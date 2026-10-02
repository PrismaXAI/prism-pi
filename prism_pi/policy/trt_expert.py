"""TensorRT engine for the pi0.5 ACTION EXPERT denoising step (policy server only).

One inference runs the 300M action expert `num_inference_steps` (10) times: v_t = f(x_t, t | prefix KV cache). With the
vision tower and the Gemma prefix already in TensorRT, those steps are the largest remaining stage (RTX 5090 laptop:
~4.4 ms per step as a bf16 CUDA-graph replay, 44 of ~77 ms). This engine is ONE denoising step with fixed shapes
(batch 1, prefix L, chunk S), called once per step so RTC guidance, which acts between steps, is untouched.

Engine interface (what PI05Pytorch.denoise_step computes, with everything that is constant over the steps as inputs):
  x_t            (1, S, A)  fp32   noisy action chunk
  time_emb       (1, W)     fp32   sinusoidal embedding of t, computed outside exactly as LeRobot does (float64)
  attention_mask (1,1,S,L+S) dt    additive mask: 0 = attend, large negative = masked prefix padding
  position_ids   (1, S)     int64
  keys, values   (layers, 1, kv_heads, L, head_dim) dt   the prefix KV cache
  -> v_t         (1, S, A)  fp32
dt = fp16 or bf16 (TensorRT 11 networks are strongly typed: the ONNX is exported in that dtype). The time MLP, the
adaptive RMS norms (computed in fp32 inside, as in PyTorch) and the output projection (fp32) are part of the engine.
With a TensorRT prefix engine of the same dtype the KV cache is read straight from that engine's output buffers
(no copy, and no rounding of an fp16 cache to bf16); otherwise it is copied/cast once per inference.

fp16 notes: the mask value is clamped to -6e4 (fp16 cannot hold OPENPI_ATTENTION_MASK_VALUE; softmax itself runs in
fp32, as in PyTorch); weights are converted from the checkpoint's fp32 values, not from the server's bf16 copy.

Pipeline (engines are GPU-specific, ONNX is not):
  python -m prism_pi.policy.trt_expert --ckpt <ckpt> --export --build --check [--precision bf16]
  export  (any GPU):     --export -> trt_cache/expert_L864_S50_fp16_<weights>.onnx (+ external weights, ~0.7 GB)
  build   (target GPU):  --build  -> trt_cache/expert_L864_S50_fp16_<weights>_<gpu>.plan
  check   --check: per-step velocity and final-action parity vs PyTorch bf16 (the path the engine replaces), NaN/Inf scan
Runtime: TRTExpert.load(prefix_pad_masks, kv, trt_prefix_or_None) once per inference, then .step(x_t, t) per step.
Enabled in the policy server with YAM_TRT_EXPERT=1 (precision: YAM_TRT_EXPERT_PRECISION=fp16|bf16).
"""
import argparse, copy, hashlib, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prism_pi.paths import TRT_CACHE as CACHE

MASK_MIN = -6.0e4
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
PRECISION = os.environ.get("YAM_TRT_EXPERT_PRECISION", "fp16")


class _ExpertStep(torch.nn.Module):
    """PI05Pytorch.denoise_step with the prefix KV cache, the additive mask and the time embedding as inputs."""

    def __init__(self, m, dt):
        super().__init__()
        # private copies in the export dtype: the live policy is not modified
        self.action_in_proj = copy.deepcopy(m.action_in_proj).float().to(dt)
        self.time_mlp_in = copy.deepcopy(m.time_mlp_in).float().to(dt)
        self.time_mlp_out = copy.deepcopy(m.time_mlp_out).float().to(dt)
        self.expert = copy.deepcopy(m.paligemma_with_expert.gemma_expert.model).float().to(dt)
        self.action_out_proj = copy.deepcopy(m.action_out_proj).float()          # fp32, like LeRobot's cast before it
        self.expert.config._attn_implementation = "eager"
        self.n_layers = len(self.expert.layers)
        self.dt = dt

    def forward(self, x_t, time_emb, attention_mask, position_ids, keys, values):
        from transformers.cache_utils import DynamicCache
        emb = self.action_in_proj(x_t.to(self.dt))
        cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb.to(self.dt)))))
        cache = DynamicCache()
        for i in range(self.n_layers):
            cache.update(keys[i], values[i], i)
        out = self.expert(inputs_embeds=emb, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_values=cache, use_cache=False, adarms_cond=cond).last_hidden_state
        return self.action_out_proj(out.float())


def _fingerprint(m) -> str:
    """Of the expert weights as the policy server holds them (bf16), whatever dtype they are in right now."""
    h = hashlib.sha1()
    ps = list(m.paligemma_with_expert.gemma_expert.model.named_parameters())[:6] + list(m.action_in_proj.named_parameters()) \
        + list(m.action_out_proj.named_parameters())
    for _, p in ps:
        h.update(p.detach().to(torch.bfloat16).float().flatten()[:256].cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def _stem(m, L: int, precision: str, fp: str | None) -> str:
    return f"expert_L{L}_S{m.config.chunk_size}_{precision}_{fp or _fingerprint(m)}"


def onnx_path(m, L: int, precision: str = PRECISION, fp: str | None = None) -> Path:
    return CACHE / (_stem(m, L, precision, fp) + ".onnx")


def plan_path(m, L: int, precision: str = PRECISION, fp: str | None = None) -> Path:
    cc = "sm%d%d" % torch.cuda.get_device_capability(0)
    return CACHE / (_stem(m, L, precision, fp) + f"_{cc}.plan")


def _kv_shape(m, L: int):
    cfg = m.paligemma_with_expert.gemma_expert.model.config
    return (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, L, cfg.head_dim)


def export_onnx(m, L: int = 864, precision: str = PRECISION, log=print, fp: str | None = None) -> Path:
    """m = policy.model (PI05Pytorch). Exports from copies of the expert modules; pass a model holding the checkpoint's
    fp32 weights so the fp16 engine is rounded once (fp32 -> fp16) rather than twice (fp32 -> bf16 -> fp16)."""
    CACHE.mkdir(exist_ok=True)
    path = onnx_path(m, L, precision, fp)
    if path.exists():
        return path
    dt = DTYPES[precision]
    mod = _ExpertStep(m, dt).to("cuda").eval()
    S, A, W = m.config.chunk_size, m.config.max_action_dim, m.action_in_proj.out_features
    args = (torch.zeros(1, S, A, device="cuda"), torch.zeros(1, W, device="cuda"),
            torch.zeros(1, 1, S, L + S, dtype=dt, device="cuda"), (L + torch.arange(S, device="cuda"))[None],
            torch.zeros(_kv_shape(m, L), dtype=dt, device="cuda"), torch.zeros(_kv_shape(m, L), dtype=dt, device="cuda"))
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(mod, args, str(path), input_names=["x_t", "time_emb", "attention_mask", "position_ids", "keys", "values"],
                          output_names=["v_t"], opset_version=18, dynamo=True, external_data=True)
    log(f"[trt_expert] ONNX exported in {time.time() - t0:.0f} s -> {path.name} ({sum(f.stat().st_size for f in CACHE.glob(path.stem + '*')) / 2**30:.2f} GB)")
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
    log(f"[trt_expert] engine built in {time.time() - t0:.0f} s -> {plan.name} ({plan.stat().st_size / 2**30:.2f} GB)")
    return plan


class TRTExpert:
    def __init__(self, plan: Path, m, L: int, precision: str = PRECISION, log=print):
        import tensorrt as trt
        from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
        self.L, self.S, self.dt = L, m.config.chunk_size, DTYPES[precision]
        self.mask_value = MASK_MIN if precision == "fp16" else OPENPI_ATTENTION_MASK_VALUE
        self.W, self.min_period, self.max_period = m.action_in_proj.out_features, m.config.min_period, m.config.max_period
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(plan).read_bytes())
        self.ctx = self.engine.create_execution_context()
        shape = lambda n: tuple(self.engine.get_tensor_shape(n))
        assert shape("attention_mask") == (1, 1, self.S, L + self.S), f"engine built for another prefix/chunk length: {shape('attention_mask')}"
        self.x = torch.zeros(shape("x_t"), dtype=torch.float32, device="cuda")
        self.temb = torch.zeros(shape("time_emb"), dtype=torch.float32, device="cuda")
        self.mask = torch.zeros(shape("attention_mask"), dtype=self.dt, device="cuda")
        self.pos = torch.zeros(shape("position_ids"), dtype=torch.int64, device="cuda")
        self.keys = torch.zeros(shape("keys"), dtype=self.dt, device="cuda")
        self.values = torch.zeros(shape("values"), dtype=self.dt, device="cuda")
        self.out = torch.zeros(shape("v_t"), dtype=torch.float32, device="cuda")
        for name, t in (("x_t", self.x), ("time_emb", self.temb), ("attention_mask", self.mask), ("position_ids", self.pos),
                        ("keys", self.keys), ("values", self.values), ("v_t", self.out)):
            self.ctx.set_tensor_address(name, t.data_ptr())
        self._arange = torch.arange(self.S, device="cuda")
        self._temb_cache = {}
        self.n_steps = 0
        self.shared_kv = False
        log(f"[trt_expert] engine {Path(plan).name}: L={L}, S={self.S}, {precision}")

    def _time_emb(self, t: float):
        e = self._temb_cache.get(t)
        if e is None:   # the same float64 computation and fp32 cast as PI05Pytorch.embed_suffix
            from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
            tt = torch.tensor([t], dtype=torch.float32, device="cuda")
            e = create_sinusoidal_pos_embedding(tt, self.W, min_period=self.min_period, max_period=self.max_period, device=tt.device).to(torch.float32)
            self._temb_cache[t] = e
        return e

    def load(self, prefix_pad_masks, past_key_values=None, trt_prefix=None):
        """Once per inference, before the steps. trt_prefix = the TRTPrefix engine that has just produced this prefix
        (its output buffers are read directly when dtype and shape match); otherwise past_key_values (a DynamicCache)
        is copied in. Builds the additive mask and the position ids of the suffix from the prefix padding mask."""
        if trt_prefix is not None and trt_prefix.dt == self.dt and tuple(trt_prefix.keys.shape) == tuple(self.keys.shape):
            k, v = trt_prefix.keys, trt_prefix.values
            self.shared_kv = True
        else:
            for i, layer in enumerate(past_key_values.layers):
                self.keys[i].copy_(layer.keys[0:1]); self.values[i].copy_(layer.values[0:1])
            k, v = self.keys, self.values
            self.shared_kv = False
        self.ctx.set_tensor_address("keys", k.data_ptr()); self.ctx.set_tensor_address("values", v.data_ptr())
        valid = prefix_pad_masks[0].to(torch.bool)
        self.mask.zero_()
        self.mask[0, 0, :, :self.L] = torch.where(valid, 0.0, self.mask_value).to(self.dt)[None, :]
        self.pos[0] = valid.sum() + self._arange

    def step(self, x_t, t: float):
        self.x.copy_(x_t[0:1])
        self.temb.copy_(self._time_emb(float(t)))
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT expert execute failed")
        self.n_steps += 1
        return self.out.clone()


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
    ap.add_argument("--task", default="Pick and place various fruits into containers", help="prompt for --check")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--precision", default=PRECISION, choices=list(DTYPES))
    a = ap.parse_args()
    pol = PI05Policy.from_pretrained(a.ckpt).to("cuda").eval()          # fp32 weights: what the engine is exported from
    for p in pol.parameters(): p.requires_grad_(False)
    m = pol.model
    fp0 = _fingerprint(m)
    onnx = export_onnx(m, a.L, a.precision, fp=fp0) if a.export else onnx_path(m, a.L, a.precision, fp0)
    plan = plan_path(m, a.L, a.precision, fp0)
    if a.build:
        build_engine(onnx, plan)
    if a.check:
        from PIL import Image
        import glob as _glob
        pol.to(torch.bfloat16)                                           # the server's PyTorch path: bf16 weights + autocast
        pre, post = make_pre_post_processors(pol.config, a.ckpt)
        for s in pre.steps:
            if type(s).__name__ == "TokenizerProcessorStep":
                s.max_length = a.prompt_tokens; s.__post_init__()
        eng = TRTExpert(plan, m, a.L, a.precision)
        rng = np.random.default_rng(0)
        highs = sorted(_glob.glob(os.path.join(a.frames, "*_high.jpg"))) if a.frames else []
        A = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; n_steps = m.config.num_inference_steps
        v_rel, v_abs, d_act, bad, t_pt, t_trt = [], [], [], 0, [], []
        for i in range(a.n):
            if a.frames:
                f = highs[(i * max(1, len(highs) // a.n)) % len(highs)]
                fr = {c: np.asarray(Image.open(f.replace("_high", f"_{c}")).convert("RGB")) for c in ("high", "left_wrist", "right_wrist")}
            else:
                fr = {c: rng.integers(0, 255, (400, 640, 3), dtype=np.uint8) for c in ("high", "left_wrist", "right_wrist")}
            st = np.array([0, 0, 0, -0.08, 0, 0, 1.45, 0, 0, 0, -0.07, 0, 0, 1.34], np.float32) + rng.normal(0, 0.1, 14).astype(np.float32) * np.r_[np.ones(6), 0, np.ones(6), 0].astype(np.float32)
            o = {f"observation.images.{c}": fr[c].copy() for c in fr}; o["observation.state"] = st
            b = pre(prepare_observation_for_inference(o, torch.device("cuda"), a.task))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                images, img_masks = pol._preprocess_images(b); tokens, masks = b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK]
                pe, ppm, pam = m.embed_prefix(images, img_masks, tokens, masks)
                a4 = m._prepare_attention_masks_4d(make_att_2d_masks(ppm, pam)); pos = torch.cumsum(ppm, 1) - 1
                m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
                _, kv = m.paligemma_with_expert.forward(attention_mask=a4, position_ids=pos, past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
                eng.load(ppm, kv)
                torch.manual_seed(i); x_ref = m.sample_noise((1, m.config.chunk_size, m.config.max_action_dim), "cuda"); x_trt = x_ref.clone()
                dt_ = -1.0 / n_steps
                for s_ in range(n_steps):
                    t_ = 1.0 + s_ * dt_; tt = torch.tensor(t_, dtype=torch.float32, device="cuda").expand(1)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    v_ref = m.denoise_step(prefix_pad_masks=ppm, past_key_values=kv, x_t=x_trt, timestep=tt).float()   # same input as the engine
                    torch.cuda.synchronize(); t_pt.append(time.perf_counter() - t0); t0 = time.perf_counter()
                    v_trt = eng.step(x_trt, t_)
                    torch.cuda.synchronize(); t_trt.append(time.perf_counter() - t0)
                    bad += int(not torch.isfinite(v_trt).all())
                    v_rel.append(float((v_trt - v_ref).norm() / v_ref.norm())); v_abs.append(float((v_trt - v_ref).abs().max()))
                    x_trt = x_trt + dt_ * v_trt
                    x_ref = x_ref + dt_ * m.denoise_step(prefix_pad_masks=ppm, past_key_values=kv, x_t=x_ref, timestep=tt).float()   # independent PyTorch trajectory
                pad = torch.zeros(1, m.config.chunk_size, 0, device="cuda")
                out_ref = post(x_ref[:, :, :14].float()).cpu().numpy()[0]; out_trt = post(x_trt[:, :, :14].float()).cpu().numpy()[0]
            bad += int(not np.isfinite(out_trt).all())
            d_act.append(np.abs(out_trt - out_ref)[:, A].max())
        print(f"per-step velocity vs PyTorch bf16 (same x_t): rel-L2 mean {np.mean(v_rel) * 100:.2f}% max {np.max(v_rel) * 100:.2f}%, max |dv| {np.max(v_abs):.4f} (normalized units) | "
              f"step time PyTorch eager {np.median(t_pt) * 1000:.1f} ms, TensorRT {np.median(t_trt) * 1000:.2f} ms on {torch.cuda.get_device_name()}")
        print(f"actions after {n_steps} steps vs PyTorch bf16: max|d| {np.max(d_act):.4f} rad (mean over obs {np.mean(d_act):.4f}); seed-to-seed spread is ~0.03-0.07 | "
              f"non-finite outputs: {bad} of {a.n * (n_steps + 1)}")
