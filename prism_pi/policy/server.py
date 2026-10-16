"""Persistent pi0.5 policy server: keeps one checkpoint loaded in VRAM and serves inference over a local Unix socket,
so executor / sim / probe runs start in ~1 s instead of paying a ~35 s model load each time.

  setsid nohup env HF_HUB_OFFLINE=1 .venv/bin/python -m prism_pi.policy.server --ckpt <checkpoint dir> > policy_server.log 2>&1 &

Protocol (multiprocessing.connection, pickle, authkey): request = dict(op=...), reply = dict(ok=bool, ...).
  ping / info                       -> what is loaded (ckpt, dtype, relative?, follower?, counters, VRAM)
  load(ckpt, dtype="bf16")          -> hot-swap the checkpoint (one model in VRAM at a time; bf16 + engines ~9 GiB of GPU, fits a 16 GB card)
  infer(frames, state, task, n_samples=1, seed=None, num_steps=None)   num_steps: flow steps (default 10)
                                    -> chunks (K, 50, 14) ABSOLUTE actions (post-processed: unnormalized, and for
                                       relative-action checkpoints converted back to absolute with the given state),
                                       norm (K, 50, 14) normalized model output, latency_s
     n_samples=K runs K independent flow-matching samples for the SAME observation in one batched forward pass
     (the hook for planning schemes: plan selection, averaging, disagreement estimates).
     rtc={"prev_abs": (H,14) ABSOLUTE actions of the running plan from the observation tick onward, "delay": d,
          "horizon": e}  -> real-time chunking: the new chunk is denoised with guidance toward that tail (weight 1
          for steps < d, linear decay to 0 at e). The tail is pushed through the checkpoint's OWN preprocessor with
          the new observation, so for relative-action checkpoints it is re-expressed as deltas from the NEW state
          and normalized exactly as in training before it is used as guidance.
  shutdown
This process never touches the robot (no i2rt / CAN imports).
"""
import argparse, json, os, threading, time, traceback
from contextlib import nullcontext
import os
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

from prism_pi.policy import relative_actions_compat  # noqa: F401  registers the fork's relative-action step name
from prism_pi.policy.client import AUTH, SOCK   # one definition of the socket path / auth key for both ends (YAM_POLICY_SOCK overrides)

PROMPT_MAX_TOKENS = 96   # see load(): prompt padding length (checkpoint default 200)
PREFIX_LEN = 3 * 256 + PROMPT_MAX_TOKENS   # prefix tokens = 3 images x 256 + prompt; the TensorRT prefix engine is built for exactly this
TRT_VISION = os.environ.get("YAM_TRT_VISION", "1") == "1"   # TensorRT SigLIP (trt_vision.py); set 0 for eager
TRT_PRECISION = os.environ.get("YAM_TRT_PRECISION", "bf16")   # bf16 = same dtype as the rest of the model (fp16 is ~2 ms faster)
TRT_PREFIX = os.environ.get("YAM_TRT_PREFIX", "1") == "1"      # TensorRT Gemma prefix (trt_prefix.py); needs the ONNX in trt_cache/
TRT_PREFIX_PRECISION = os.environ.get("YAM_TRT_PREFIX_PRECISION", "fp16")   # fp16 | bf16 (each needs its own export)
TRT_EXPERT = os.environ.get("YAM_TRT_EXPERT", "0") == "1"      # TensorRT action expert (trt_expert.py), opt-in; needs the ONNX in trt_cache/
TRT_EXPERT_PRECISION = os.environ.get("YAM_TRT_EXPERT_PRECISION", "fp16")   # fp16 | bf16 (each needs its own export)
# VRAM (2026-10-15, so that a 16 GB card such as an RTX 5070 Ti fits): the model is cast to bf16 and stripped of its two
# unused LM heads on the host before anything touches the GPU, engines are built while the GPU is still empty, and the
# PyTorch twin of every engine that passed the parity gate (Gemma prefix layers, SigLIP) is parked in host RAM
# (YAM_OFFLOAD_REPLACED=0 keeps them resident; they come back on demand, see fast_denoise.GraphedDenoiser.restore).
OFFLOAD_REPLACED = os.environ.get("YAM_OFFLOAD_REPLACED", "1") == "1"
IMG_KEYS = ("high", "left_wrist", "right_wrist")


def gpu_mem_gib():
    """(used, total) of the whole device in GiB, as nvidia-smi reports it: CUDA context + torch + TensorRT, every process."""
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 2**30, 2), round(total / 2**30, 2)


class PolicyHost:
    def __init__(self):
        self.lock = threading.Lock()
        self.policy = None
        self.meta = {"ckpt": None}

    def load(self, ckpt: str, dtype: str = "bf16"):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.utils import prepare_observation_for_inference
        from transformers import AutoTokenizer

        ckpt = str(Path(ckpt).resolve())
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        ids = tok("Pick and place various fruits into containers", add_special_tokens=True)["input_ids"]
        if not ids or ids[0] != 2:
            raise RuntimeError("prompt tokenization lacks <bos> (id 2): substitute tokenizer in the HF cache")
        with self.lock:
            if self.policy is not None:
                # release everything that pins GPU memory: CUDA graph + static buffers (fast_denoise), the TensorRT
                # engine/context, then the model; otherwise the next load OOMs (a 16 GB card has no room for two)
                fast = getattr(self, "fast", None)
                if fast is not None:
                    fast.graph = None; fast.s_cache = None; fast.s_out = None; fast.s_x = None
                    fast.trt = None; fast.trt_lm = None; fast.trt_expert = None
                    fast.offloaded = {}; fast.policy = None; fast.m = None
                    self.fast = None
                del fast   # the local would keep the old model alive (fast.policy) until this call returns
                del self.policy, self.pre, self.post
                self.policy = None
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print(f"[policy_server] released, VRAM now {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
            log = lambda m: print(m, flush=True)   # noqa: E731
            from prism_pi.policy import fast_denoise
            # bf16 WEIGHTS (not just autocast): -27 ms per inference (189 -> 163 ms), max |Δ| 0.003 rad (2026-09-21).
            # Built, filled and cast in host RAM with the unused LM heads dropped: the fp32 model never reaches the GPU.
            self.policy = fast_denoise.load_policy_on_host(ckpt, bf16=(dtype == "bf16"), log=log)
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=16)   # no leftover -> plain denoising
            self.policy.init_rtc_processor()
            plans = self._build_engines(log)     # export / build whatever plan is missing while the GPU is still empty
            self.policy.to("cuda")
            self.pre, self.post = make_pre_post_processors(self.policy.config, ckpt)
            self._prep = self._prep_gpu   # lerobot's prepare_observation_for_inference converts images on the CPU: 52 ms for 3 frames
            # CUDA-graph replay of the denoising step (fast_denoise.py): 10 steps 86 -> ~17 ms, bit-identical output;
            # also sets requires_grad=False on the weights (RTC guidance no longer records a graph: -35 ms).
            self.fast = fast_denoise.install(self.policy)
            if "vision" in plans:
                # TensorRT SigLIP (all 3 images in one engine call): image embedding 18 -> 8 ms (bf16; fp16 6 ms). Not bit-identical
                # (max |Δ| 0.005 rad on actions, vs 0.033 rad seed-to-seed spread): parity is re-checked after warm-up.
                try:
                    from prism_pi.policy import trt_vision
                    self.fast.trt = trt_vision.TRTImageEncoder(self.policy.model.paligemma_with_expert, 3, TRT_PRECISION, log=log)
                except Exception as e:
                    print(f"[policy_server] TensorRT vision unavailable, eager SigLIP: {e!r}", flush=True)
                    self.fast.trt = None
            if "prefix" in plans:
                # TensorRT fp16 Gemma prefix: fp16-accumulate GEMMs run at full rate on this GPU (bf16 at half rate).
                try:
                    from prism_pi.policy import trt_prefix
                    lm_ = self.policy.model.paligemma_with_expert.paligemma.model.language_model
                    self.fast.trt_lm = trt_prefix.TRTPrefix(plans["prefix"], PREFIX_LEN, lm_.config.num_hidden_layers, log=log, precision=TRT_PREFIX_PRECISION)
                except Exception as e:
                    print(f"[policy_server] TensorRT prefix unavailable, eager Gemma: {e!r}", flush=True)
                    self.fast.trt_lm = None
            if "expert" in plans:
                # TensorRT action expert: one engine call per denoising step instead of a CUDA-graph replay of the bf16
                # PyTorch step (RTX 5090 laptop: 4.4 -> 2.4 ms per step in fp16). With an fp16 prefix engine the fp16
                # expert reads the KV cache straight from its output buffers.
                try:
                    from prism_pi.policy import trt_expert
                    self.fast.trt_expert = trt_expert.TRTExpert(plans["expert"], self.policy.model, PREFIX_LEN, TRT_EXPERT_PRECISION, log=log)
                except Exception as e:
                    print(f"[policy_server] TensorRT expert unavailable, CUDA-graph PyTorch expert: {e!r}", flush=True)
                    self.fast.trt_expert = None
            # Prompt padding: the checkpoint pads the prompt to 200 tokens, the real prompt (task + discretized state)
            # is ~60 → 140 masked pad tokens in every prefix. 96 keeps a margin for all task strings/states: -5 ms,
            # max |Δ| 0.003 rad (masked positions only change bf16 reduction order). Guarded against truncation below.
            for st_ in self.pre.steps:
                if type(st_).__name__ == "TokenizerProcessorStep" and getattr(st_, "max_length", 0) > PROMPT_MAX_TOKENS:
                    st_.max_length = PROMPT_MAX_TOKENS
                    try:
                        st_.__post_init__()
                    except Exception:
                        pass
            self.autocast = dtype == "bf16"
            cfg = json.loads((Path(ckpt) / "config.json").read_text())
            follower = (Path(ckpt) / "FOLLOWER_ACTIONS").exists()
            try:
                follower |= "follower" in (Path(ckpt) / "train_config.json").read_text().lower()
            except Exception:
                pass
            self.meta = {"ckpt": ckpt, "dtype": dtype, "relative": bool(cfg.get("use_relative_actions", False)),
                         "follower": follower, "loaded_at": time.time(), "n_infer": 0}
        # warm-up (CUDA init / graph capture / autotune) so the first real request is already fast
        rng = np.random.default_rng(0)
        z = {k: rng.integers(0, 255, (400, 640, 3), dtype=np.uint8) for k in IMG_KEYS}
        s = np.array([0, 0, 0, -0.08, 0, 0, 1.45, 0, 0, 0, -0.07, 0, 0, 1.34], np.float32)
        for _ in range(2):
            self.infer(z, s, "Pick and place various fruits into containers")
        # parity checks of the TensorRT stages against eager PyTorch (same seed, 3 random observations): each stage is
        # kept only if the actions it produces are finite and stay well inside the seed-to-seed spread (~0.03-0.07 rad)
        A = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        obs_chk = []
        for k in range(3):
            zz = {kk: rng.integers(0, 255, (400, 640, 3), dtype=np.uint8) for kk in IMG_KEYS}
            ss = s + rng.normal(0, 0.05, 14).astype(np.float32) * np.r_[np.ones(6), 0, np.ones(6), 0].astype(np.float32)
            obs_chk.append((zz, ss))
        def _acts():
            return [self.infer(zz, ss, "Pick and place various fruits into containers", seed=0, check_finite=False)["chunks"][0] for zz, ss in obs_chk]
        for stage, attr, thr in (("vision", "trt", 0.02), ("prefix", "trt_lm", 0.03), ("expert", "trt_expert", 0.03)):
            eng = getattr(self.fast, attr, None)
            if eng is None:
                continue
            with_ = _acts(); setattr(self.fast, attr, None); without = _acts()
            d = float(np.mean([np.abs(a_ - b_)[:, A].max() for a_, b_ in zip(with_, without)]))   # NaN/Inf in either -> not < thr -> OFF
            if d < thr:
                setattr(self.fast, attr, eng)
                print(f"[policy_server] TensorRT {stage} ON (parity max |Δ| {d:.4f} rad, mean over 3 obs)", flush=True)
            else:
                print(f"[policy_server] TensorRT {stage} OFF: parity max |Δ| {d:.4f} rad", flush=True)
        if OFFLOAD_REPLACED:
            self._offload_replaced()
        self.meta["n_infer"] = 0
        self.meta["load_s"] = round(time.time() - t0, 1)
        self.meta["vram_gib"] = round(torch.cuda.memory_allocated() / 2**30, 1)   # torch tensors only (engines are outside the allocator)
        self.meta["gpu_used_gib"], self.meta["gpu_total_gib"] = gpu_mem_gib()
        print(f"[policy_server] GPU in use {self.meta['gpu_used_gib']} of {self.meta['gpu_total_gib']} GiB (torch {self.meta['vram_gib']} GiB)", flush=True)
        return dict(self.meta)

    def _build_engines(self, log) -> dict:
        """Export / build the plans of the enabled TensorRT stages that are not cached yet, BEFORE the model is moved to
        the GPU (the builder wants up to 8 GiB of workspace plus the weights, which does not fit next to a resident
        model on a 16 GB card). Returns {stage: plan path}; a stage that fails here is reported and left eager."""
        plans = {}
        pwe = self.policy.model.paligemma_with_expert
        if TRT_VISION:
            try:
                from prism_pi.policy import trt_vision
                plans["vision"] = trt_vision.build_engine(pwe, 3, TRT_PRECISION, log=log)   # exports through a temporary GPU copy of SigLIP
            except Exception as e:
                print(f"[policy_server] TensorRT vision unavailable, eager SigLIP: {e!r}", flush=True)
        if TRT_PREFIX:
            # ONNX exported once per checkpoint (python -m prism_pi.policy.trt_prefix --export --build); the engine is built here per GPU.
            try:
                from prism_pi.policy import trt_prefix
                lm_ = pwe.paligemma.model.language_model
                onnx_ = trt_prefix.onnx_path(lm_, PREFIX_LEN, TRT_PREFIX_PRECISION)
                if not onnx_.exists():
                    raise FileNotFoundError(f"no exported ONNX {onnx_.name} (run python -m prism_pi.policy.trt_prefix --export --build for this checkpoint)")
                plans["prefix"] = trt_prefix.build_engine(onnx_, trt_prefix.plan_path(lm_, PREFIX_LEN, TRT_PREFIX_PRECISION), log=log)
            except Exception as e:
                print(f"[policy_server] TensorRT prefix unavailable, eager Gemma: {e!r}", flush=True)
        if TRT_EXPERT:
            # ONNX exported once per checkpoint (python -m prism_pi.policy.trt_expert --export --build); the engine is built here per GPU.
            try:
                from prism_pi.policy import trt_expert
                m_ = self.policy.model
                onnx_ = trt_expert.onnx_path(m_, PREFIX_LEN, TRT_EXPERT_PRECISION)
                if not onnx_.exists():
                    raise FileNotFoundError(f"no exported ONNX {onnx_.name} (run python -m prism_pi.policy.trt_expert --export --build for this checkpoint)")
                plans["expert"] = trt_expert.build_engine(onnx_, trt_expert.plan_path(m_, PREFIX_LEN, TRT_EXPERT_PRECISION), log=log)
            except Exception as e:
                print(f"[policy_server] TensorRT expert unavailable, CUDA-graph PyTorch expert: {e!r}", flush=True)
        return plans

    def _offload_replaced(self):
        """Park the PyTorch modules whose work a confirmed engine now does in host RAM: the 18 Gemma prefix layers
        (3.9 GiB bf16) and SigLIP + projector (0.8 GiB). The token embedding stays (the prompt is embedded in PyTorch)
        and so does the action expert (the runtime NaN guard falls back to it). fast_denoise brings a stage back to the
        GPU by itself when it needs the eager path (engine failure, or a batched K > 1 request)."""
        pg = self.policy.model.paligemma_with_expert.paligemma
        off = {}
        if self.fast.trt_lm is not None:
            off["prefix"] = [pg.model.language_model.layers, pg.model.language_model.norm]
        if self.fast.trt is not None:
            off["vision"] = [pg.model.vision_tower, pg.model.multi_modal_projector]
        if not off:
            return
        before = torch.cuda.memory_allocated()
        for mods in off.values():
            for mod in mods:
                mod.to("cpu")
        self.fast.offloaded = off
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[policy_server] parked in host RAM: {', '.join(off)} ({(before - torch.cuda.memory_allocated()) / 2**30:.1f} GiB freed)", flush=True)

    @staticmethod
    def _prep_gpu(observation: dict, device, task: str):
        """Same output as lerobot.policies.utils.prepare_observation_for_inference, images converted ON the GPU."""
        out = {}
        for name, v in observation.items():
            t = torch.from_numpy(np.ascontiguousarray(v))
            if "image" in name:
                t = t.to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0)
            else:
                t = t.to(device)
            out[name] = t.unsqueeze(0)
        out["task"] = task if task else ""
        out["robot_type"] = ""
        return out

    @torch.no_grad()   # not inference_mode: RTC guidance re-enables autograd internally
    def infer(self, frames: dict, state, task: str, n_samples: int = 1, seed=None, rtc=None, num_steps=None, check_finite=True):
        if self.policy is None:
            raise RuntimeError("no checkpoint loaded")
        t0 = time.perf_counter()
        with self.lock:
            if seed is not None:
                torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
            obs = {f"observation.images.{k}": np.ascontiguousarray(frames[k]) for k in IMG_KEYS}
            obs["observation.state"] = np.asarray(state, np.float32)
            b_in = self._prep(obs, torch.device("cuda"), task)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            kw = {}
            if rtc is not None and len(rtc["prev_abs"]) >= 2:
                b_in["action"] = torch.as_tensor(np.asarray(rtc["prev_abs"], np.float32), device="cuda")[None]   # (1,H,14) absolute
            batch = self.pre(b_in)
            _m = batch.get("observation.language.attention_mask")
            if _m is not None and bool(_m.all()):   # every slot used -> the prompt may have been truncated
                raise RuntimeError(f"prompt fills all {int(_m.shape[-1])} token slots (possible truncation): raise PROMPT_MAX_TOKENS")
            torch.cuda.synchronize(); t2 = time.perf_counter()
            k = max(1, int(n_samples))
            if "action" in b_in and batch.get("action") is not None:
                left = batch["action"].float()                       # (1,H,14): relative-converted (if applicable) + normalized
                kw = dict(inference_delay=int(rtc["delay"]), execution_horizon=int(rtc.get("horizon", 16)),
                          prev_chunk_left_over=left.expand(k, -1, -1).contiguous())
                batch = {key: v for key, v in batch.items() if key != "action"}
            if k > 1:  # same observation K times -> K independent noise samples in one forward pass
                batch = {key: (v.repeat_interleave(k, dim=0) if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == 1 else v)
                         for key, v in batch.items()}
            if num_steps:   # flow-matching denoising steps (checkpoint default 10); fewer = lower latency
                kw["num_steps"] = int(num_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16) if self.autocast else nullcontext():
                norm = self.policy.predict_action_chunk(batch, **kw)     # (K, 50, 14) normalized
            torch.cuda.synchronize(); t3 = time.perf_counter()
            chunks = self.post(norm.float()).float().cpu().numpy()      # absolute actions
            if check_finite and not np.isfinite(chunks).all():
                # never hand NaN/Inf targets to a controller: the request fails (the executor parks on a failed inference)
                # and a TensorRT expert, the stage most exposed to fp16 overflow, is dropped for the following requests
                if getattr(self.fast, "trt_expert", None) is not None:
                    self.fast.trt_expert = None
                    print("[policy_server] non-finite actions: TensorRT expert disabled, back to the PyTorch expert", flush=True)
                raise RuntimeError("model produced non-finite actions")
            self.meta["n_infer"] += 1
        return {"chunks": chunks, "norm": norm.float().cpu().numpy(), "latency_s": time.perf_counter() - t0,
                "timing_ms": {"prep": (t1 - t0) * 1000, "pre": (t2 - t1) * 1000, "model": (t3 - t2) * 1000, "post": (time.perf_counter() - t3) * 1000}}


def serve(host: PolicyHost):
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    listener = Listener(address=SOCK, family="AF_UNIX", authkey=AUTH)
    print(f"[policy_server] listening on {SOCK}", flush=True)
    stop = threading.Event()

    def handle(conn):
        try:
            while not stop.is_set():
                try:
                    req = conn.recv()
                except (EOFError, ConnectionResetError):
                    return
                op = req.get("op")
                try:
                    if op in ("ping", "info"):
                        rep = {"ok": True, **host.meta, "vram_gib_now": round(torch.cuda.memory_allocated() / 2**30, 1), "gpu_used_gib_now": gpu_mem_gib()[0]}
                    elif op == "load":
                        print(f"[policy_server] loading {req['ckpt']} ({req.get('dtype', 'bf16')})", flush=True)
                        rep = {"ok": True, **host.load(req["ckpt"], req.get("dtype", "bf16"))}
                        print(f"[policy_server] loaded in {rep['load_s']} s, {rep['vram_gib']} GiB", flush=True)
                    elif op == "infer":
                        rep = {"ok": True, **host.infer(req["frames"], req["state"], req["task"], req.get("n_samples", 1), req.get("seed"), req.get("rtc"), req.get("num_steps"))}
                    elif op == "shutdown":
                        conn.send({"ok": True}); stop.set()
                        try:
                            listener.close()
                        finally:
                            os._exit(0)
                    else:
                        rep = {"ok": False, "error": f"unknown op {op!r}"}
                except Exception as e:  # report, keep serving
                    rep = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}
                    print("[policy_server] request failed:", rep["traceback"], flush=True)
                conn.send(rep)
        finally:
            conn.close()

    while not stop.is_set():
        try:
            conn = listener.accept()
        except OSError:
            break
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LeRobot pi0.5 checkpoint directory (or a shim, tools/make_ckpt_shim.py)")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    a = ap.parse_args()
    h = PolicyHost()
    m = h.load(a.ckpt, a.dtype)
    print(f"[policy_server] ready: {m}", flush=True)
    serve(h)
