"""TensorRT engine for the pi0.5 Gemma PREFIX pass (policy server only).

The prefix pass runs the 18 Gemma layers once per inference over the 864 prefix tokens (3 x 256 image tokens + 96 prompt
tokens) and produces the KV cache the action expert attends to. It is compute-bound: ~52 ms in PyTorch bf16 on the RTX
5090 laptop, whose tensor cores run fp32-accumulate ops (bf16, and fp16 in cuBLAS) at half rate. A TensorRT fp16 engine
uses fp16-accumulate GEMMs and fused attention.

Pipeline (engines are GPU-specific, ONNX is not):
  export  (any GPU):  --export  -> trt_cache/prefix_L864_fp16.onnx (+ external weights)   [wrapper around language_model.forward]
  build   (target GPU): --build -> trt_cache/prefix_L864_fp16_<gpu>.plan
  check   (any GPU with the checkpoint): --check -> KV parity vs PyTorch bf16 and action parity (engine KV fed to the expert)
Runtime: TRTPrefix(...)(inputs_embeds (1,L,2048), attention_mask_4d (1,1,L,L) additive, position_ids (1,L)) -> DynamicCache
Mask: fp16 cannot hold OPENPI_ATTENTION_MASK_VALUE (-2.4e38); the additive mask is clamped to -6e4 (fully masked pad rows
still get a finite, uniform softmax, as in bf16).
"""
import argparse, hashlib, os, sys, time
from pathlib import Path

import numpy as np
import torch

from prism_pi.paths import TRT_CACHE as CACHE

MASK_MIN = -6.0e4
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
PRECISION = os.environ.get("YAM_TRT_PREFIX_PRECISION", "fp16")


class _PrefixKV(torch.nn.Module):
    def __init__(self, lm, clamp_mask: bool):
        super().__init__()
        self.lm = lm; self.clamp_mask = clamp_mask

    def forward(self, inputs_embeds, attention_mask, position_ids):
        if self.clamp_mask:
            attention_mask = attention_mask.clamp(min=MASK_MIN)
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
        kv = out.past_key_values
        keys = torch.stack([l.keys for l in kv.layers], 0)
        values = torch.stack([l.values for l in kv.layers], 0)
        return keys, values


def _fingerprint(lm) -> str:
    h = hashlib.sha1()
    for n, p in list(lm.named_parameters())[:6]:
        h.update(p.detach().float().flatten()[:256].cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def onnx_path(lm, L: int, precision: str = PRECISION, fp: str | None = None) -> Path:
    return CACHE / f"prefix_L{L}_{precision}_{fp or _fingerprint(lm)}.onnx"


def plan_path(lm, L: int, precision: str = PRECISION, fp: str | None = None) -> Path:
    cc = "sm%d%d" % torch.cuda.get_device_capability(0)
    return CACHE / f"prefix_L{L}_{precision}_{fp or _fingerprint(lm)}_{cc}.plan"


def export_onnx(policy, L: int = 864, precision: str = PRECISION, log=print, fp: str | None = None) -> Path:
    """Exports the language model (converted to the export dtype IN PLACE; a bf16 -> fp16 -> bf16 round trip can change
    small weights, so the weight fingerprint is taken BEFORE the conversion and passed as `fp` for later path lookups)."""
    CACHE.mkdir(exist_ok=True)
    lm = policy.model.paligemma_with_expert.paligemma.model.language_model
    path = onnx_path(lm, L, precision, fp)
    if path.exists():
        return path
    dt = DTYPES[precision]
    lm.to(dt).eval()
    mod = _PrefixKV(lm, clamp_mask=(precision == "fp16")).eval()
    D = lm.config.hidden_size
    x = torch.zeros(1, L, D, dtype=dt, device="cuda")
    mask = torch.zeros(1, 1, L, L, dtype=dt, device="cuda")
    pos = torch.arange(L, device="cuda")[None]
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(mod, (x, mask, pos), str(path), input_names=["inputs_embeds", "attention_mask", "position_ids"],
                          output_names=["keys", "values"], opset_version=18, dynamo=True, external_data=True)
    log(f"[trt_prefix] ONNX exported in {time.time() - t0:.0f} s -> {path.name} ({sum(f.stat().st_size for f in CACHE.glob(path.stem + '*')) / 2**30:.1f} GB)")
    return path


def build_engine(onnx: Path, plan: Path, log=print) -> Path:
    import tensorrt as trt
    if plan.exists():
        return plan
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t0 = time.time()
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    plan.write_bytes(bytes(blob))
    log(f"[trt_prefix] engine built in {time.time() - t0:.0f} s -> {plan.name} ({plan.stat().st_size / 2**30:.1f} GB)")
    return plan


class TRTPrefix:
    def __init__(self, plan: Path, L: int, n_layers: int, log=print, precision: str = PRECISION):
        import tensorrt as trt
        self.L = L; self.dt = DTYPES[precision]; self.clamp = precision == "fp16"
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(plan).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.x = torch.zeros(1, L, self.engine.get_tensor_shape("inputs_embeds")[-1], dtype=self.dt, device="cuda")
        self.mask = torch.zeros(1, 1, L, L, dtype=self.dt, device="cuda")
        self.pos = torch.zeros(1, L, dtype=torch.int64, device="cuda")
        kshape = tuple(self.engine.get_tensor_shape("keys"))
        self.keys = torch.zeros(kshape, dtype=self.dt, device="cuda")
        self.values = torch.zeros(kshape, dtype=self.dt, device="cuda")
        for name, t in (("inputs_embeds", self.x), ("attention_mask", self.mask), ("position_ids", self.pos), ("keys", self.keys), ("values", self.values)):
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.n_calls = 0
        log(f"[trt_prefix] engine {Path(plan).name}: L={L}, keys {kshape}")

    def __call__(self, inputs_embeds, attention_mask_4d, position_ids, out_dtype=torch.bfloat16):
        from transformers.cache_utils import DynamicCache
        assert inputs_embeds.shape[1] == self.L, f"engine built for L={self.L}, got {inputs_embeds.shape[1]}"
        self.x.copy_(inputs_embeds[0:1])
        self.mask.copy_(attention_mask_4d[0:1].clamp(min=MASK_MIN) if self.clamp else attention_mask_4d[0:1])
        self.pos.copy_(position_ids[0:1])
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT prefix execute failed")
        self.n_calls += 1
        cache = DynamicCache()
        for i in range(self.keys.shape[0]):
            cache.update(self.keys[i].to(out_dtype), self.values[i].to(out_dtype), i)
        return cache


# ---------------------------------------------------------------- CLI: export / build / check
if __name__ == "__main__":
    from prism_pi.policy import relative_actions_compat  # noqa
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--L", type=int, default=864); ap.add_argument("--prompt-tokens", type=int, default=96)
    ap.add_argument("--export", action="store_true"); ap.add_argument("--build", action="store_true"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--frames", default=None, help="dir with NNNN_{high,left_wrist,right_wrist}.jpg for --check (default: random images)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--precision", default=PRECISION, choices=list(DTYPES))
    a = ap.parse_args(); PRECISION = a.precision
    pol = PI05Policy.from_pretrained(a.ckpt).to("cuda").eval().to(torch.bfloat16)
    for p in pol.parameters(): p.requires_grad_(False)
    pre, post = make_pre_post_processors(pol.config, a.ckpt)
    for s in pre.steps:
        if type(s).__name__ == "TokenizerProcessorStep":
            s.max_length = a.prompt_tokens; s.__post_init__()
    m = pol.model; pwe = m.paligemma_with_expert; lm = pwe.paligemma.model.language_model
    fp0 = _fingerprint(lm)   # of the checkpoint's bf16 weights, as the policy server will compute it
    if a.export:
        onnx = export_onnx(pol, a.L, a.precision, fp=fp0); lm.to(torch.bfloat16)
    else:
        onnx = onnx_path(lm, a.L, a.precision, fp0)
    plan = plan_path(lm, a.L, a.precision, fp0)
    if a.build:
        build_engine(onnx, plan)
    if a.check:
        from PIL import Image
        rng = np.random.default_rng(0)
        obs_list = []
        import glob as _glob
        highs = sorted(_glob.glob(os.path.join(a.frames, "*_high.jpg"))) if a.frames else []
        for i in range(a.n):
            if a.frames:
                f = highs[i % len(highs)]
                fr = {c: np.asarray(Image.open(f.replace("_high", f"_{c}")).convert("RGB")) for c in ("high", "left_wrist", "right_wrist")}
            else:
                fr = {c: rng.integers(0, 255, (400, 640, 3), dtype=np.uint8) for c in ("high", "left_wrist", "right_wrist")}
            st = np.array([0, 0, 0, -0.08, 0, 0, 1.45, 0, 0, 0, -0.07, 0, 0, 1.34], np.float32) + rng.normal(0, 0.1, 14).astype(np.float32) * np.r_[np.ones(6), 0, np.ones(6), 0]
            obs_list.append((fr, st))
        eng = TRTPrefix(plan, a.L, lm.config.num_hidden_layers, precision=a.precision)
        def batch(fr, st):
            o = {f"observation.images.{c}": fr[c].copy() for c in fr}; o["observation.state"] = st
            return pre(prepare_observation_for_inference(o, torch.device("cuda"), "Pick and place various fruits into containers"))
        # KV parity
        rel = []; t_pt = []; t_trt = []
        for fr, st in obs_list:
            b = batch(fr, st)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                images, img_masks = pol._preprocess_images(b); tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
                pe, ppm, pam = m.embed_prefix(images, img_masks, tokens, masks)
                a4 = m._prepare_attention_masks_4d(make_att_2d_masks(ppm, pam)); pos = torch.cumsum(ppm, 1) - 1
                lm.config._attn_implementation = "eager"
                torch.cuda.synchronize(); t0 = time.perf_counter()
                _, kv = pwe.forward(attention_mask=a4, position_ids=pos, past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
                torch.cuda.synchronize(); t_pt.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); kv2 = eng(pe, a4, pos); torch.cuda.synchronize(); t_trt.append(time.perf_counter() - t0)
            rel.append(np.mean([float((kv.layers[i].keys.float() - kv2.layers[i].keys.float()).norm() / kv.layers[i].keys.float().norm()) for i in range(len(kv.layers))]))
            bad = sum(int(torch.isnan(kv2.layers[i].keys).any()) for i in range(len(kv2.layers)))
        print(f"KV rel-L2 vs PyTorch bf16: mean {np.mean(rel)*100:.1f}% (fp16 PyTorch was ~28%) | NaN layers {bad} | prefix time PyTorch {np.median(t_pt)*1000:.1f} ms, TensorRT {np.median(t_trt)*1000:.1f} ms on {torch.cuda.get_device_name(0)}")
        # action parity: engine KV fed to the expert
        A = [0,1,2,3,4,5,7,8,9,10,11,12]; orig = pwe.forward
        def fwd(*args, **k):
            ie = k.get("inputs_embeds")
            if ie is not None and ie[0] is not None and ie[1] is None:
                return None, eng(ie[0], k["attention_mask"], k["position_ids"])
            return orig(*args, **k)
        d = []
        for fr, st in obs_list:
            b = batch(fr, st)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pwe.forward = orig; torch.manual_seed(0); ref = post(pol.predict_action_chunk(b).float()).cpu().numpy()[0]
                pwe.forward = fwd; torch.manual_seed(0); out = post(pol.predict_action_chunk(b).float()).cpu().numpy()[0]
            d.append(np.abs(out - ref)[:, A].max())
        pwe.forward = orig
        print(f"actions with engine KV vs PyTorch: max|Δ| {max(d):.4f} rad (mean over obs {np.mean(d):.4f}); seed-to-seed spread is ~0.03–0.07")
