"""TensorRT engine for the pi0.5 image encoder (SigLIP vision tower + multi-modal projector), policy server only.

The three camera images are encoded in ONE batched engine call instead of three eager SigLIP passes.
Build: ONNX export of `paligemma.model.get_image_features` (+ the sqrt(hidden) scale) for a fixed batch of 3
224x224 images, then a TensorRT plan cached under trt_cache/ (keyed on checkpoint weights + precision; ~1-3 min
to build once). Runtime: `TRTImageEncoder.__call__(images: list[(1,3,224,224) fp32]) -> list[(1,256,2048)]`,
executed on the current torch stream with torch-owned buffers (no torch-tensorrt needed).
Precision: TensorRT 11 networks are strongly typed (no FP16/BF16 builder flags): the ONNX is exported in the requested
dtype (fp16 / bf16 / fp32) and the engine runs in it; inputs are cast at the boundary, outputs returned as fp32.
bf16 needs the dynamo exporter (+ onnxscript); the legacy exporter fails on bf16 constants. Measured 2026-09-22 (3 imgs):
eager bf16 17.7 ms, TRT bf16 8.4 ms, TRT fp16 6.3 ms; action parity vs eager: bf16 0.003 rad, fp16 0.007 rad.
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

CACHE = Path(__file__).resolve().parent / "trt_cache"


class _ImageEncoder(torch.nn.Module):
    def __init__(self, pwe):
        super().__init__()
        self.pg = pwe.paligemma
        self.scale = float(pwe.paligemma.config.text_config.hidden_size ** 0.5)

    def forward(self, pixel_values):
        feats = self.pg.model.get_image_features(pixel_values).pooler_output
        return feats * self.scale


def _key(pwe, precision: str) -> str:
    h = hashlib.sha1()
    for n, p in list(pwe.paligemma.model.vision_tower.named_parameters())[:8]:   # cheap fingerprint of the weights
        h.update(p.detach().float().flatten()[:256].cpu().numpy().tobytes())
    for n, p in list(pwe.paligemma.model.multi_modal_projector.named_parameters())[:2]:
        h.update(p.detach().float().flatten()[:256].cpu().numpy().tobytes())
    h.update(precision.encode())
    return h.hexdigest()[:12]


def build_engine(pwe, batch: int = 3, precision: str = "fp16", log=print) -> Path:
    import tensorrt as trt
    CACHE.mkdir(exist_ok=True)
    plan = CACHE / f"siglip_b{batch}_{precision}_{_key(pwe, precision)}.plan"
    if plan.exists():
        return plan
    onnx_path = plan.with_suffix(".onnx")
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    vt, proj = pwe.paligemma.model.vision_tower, pwe.paligemma.model.multi_modal_projector
    orig_dt = next(vt.parameters()).dtype
    vt.float(); proj.float()          # export from fp32 weights, in the requested dtype; restore the model afterwards
    enc = _ImageEncoder(pwe).to(dt).eval()
    x = torch.zeros(batch, 3, 224, 224, device="cuda", dtype=dt)
    t0 = time.time()
    try:
        with torch.no_grad():
            if precision == "bf16":   # the legacy (TorchScript) exporter cannot emit bf16 ("ComplexDouble" error); dynamo can
                torch.onnx.export(enc, (x,), str(onnx_path), input_names=["pixel_values"], output_names=["features"],
                                  opset_version=18, dynamo=True, external_data=False)
            else:
                torch.onnx.export(enc, (x,), str(onnx_path), input_names=["pixel_values"], output_names=["features"],
                                  opset_version=18, dynamo=False, do_constant_folding=True)
    finally:
        vt.to(orig_dt); proj.to(orig_dt)
    log(f"[trt_vision] ONNX exported in {time.time() - t0:.0f} s")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)   # TensorRT >= 10: explicit batch is the only mode
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    t0 = time.time()
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    plan.write_bytes(bytes(blob))
    onnx_path.unlink(missing_ok=True)
    log(f"[trt_vision] engine built in {time.time() - t0:.0f} s -> {plan.name} ({plan.stat().st_size / 2**20:.0f} MB)")
    return plan


class TRTImageEncoder:
    def __init__(self, pwe, batch: int = 3, precision: str = "fp16", log=print):
        import tensorrt as trt
        self.batch = batch
        plan = build_engine(pwe, batch, precision, log)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        self.ctx = self.engine.create_execution_context()
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        self.inp = torch.zeros(batch, 3, 224, 224, device="cuda", dtype=dt)
        out_shape = tuple(self.engine.get_tensor_shape("features"))
        self.out = torch.zeros(out_shape, device="cuda", dtype=dt)
        self.ctx.set_tensor_address("pixel_values", self.inp.data_ptr())
        self.ctx.set_tensor_address("features", self.out.data_ptr())
        self.n_calls = 0

    def __del__(self):
        for a in ("ctx", "engine", "runtime"):
            if hasattr(self, a):
                setattr(self, a, None)

    def __call__(self, images: list) -> list:
        assert len(images) == self.batch, f"engine built for {self.batch} images, got {len(images)}"
        for i, im in enumerate(images):
            self.inp[i].copy_(im[0])
        ok = self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute failed")
        self.n_calls += 1
        return [self.out[i:i + 1].float() for i in range(self.batch)]
