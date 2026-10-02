"""CUDA-graph replay of the pi0.5 denoising step (policy server only; the model file is untouched).

Why: one denoising step of the 300M action expert is ~1 ms of GPU work but ~8.6 ms wall time in eager mode
(deepcopy of the 18-layer KV cache, per-layer cache concatenation, hundreds of small kernel launches). Ten steps
per inference = ~86 ms of mostly overhead. A captured CUDA graph replays the whole step's kernel sequence in one go.

How: `GraphedDenoiser.sample_actions` mirrors `PI05FlowMatching.sample_actions` (prefix forward once, then the
step loop with the RTC processor) but the step is `denoise_step` captured once for batch size 1 with STATIC
buffers: the prefix KV cache is copied into fixed tensors after every prefix forward, x_t / timestep are copied
in before each replay. Any shape or capture problem falls back to the eager path. Works with RTC guidance because
LeRobot's guidance only back-propagates through `x1_t = x_t - t * v_t` (v_t is computed before x_t requires
grad), so no autograd through the network is needed; parameters are set to requires_grad=False for the same reason.
Verified 2026-09-21: max |Δ| vs eager 0 rad (same seed), step 8.6 -> ~1.7 ms.
"""
import contextlib

import torch
from transformers.cache_utils import DynamicCache

import math

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks


@contextlib.contextmanager
def _const_tensors():
    """embed_suffix builds its attention-mask constant with torch.tensor(<python list>) -> a synchronous H2D copy,
    which is illegal inside CUDA-graph capture. Serve such constants from a cache (created outside capture)."""
    cache = {}
    real = torch.tensor

    def fake(data, *a, **k):
        if isinstance(data, (list, tuple)) and str(k.get("device", "")).startswith("cuda"):
            key = (tuple(data), str(k.get("dtype")), str(k.get("device")))
            if key not in cache:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("constant not pre-cached before capture")
                cache[key] = real(data, *a, **k)
            return cache[key]
        return real(data, *a, **k)

    torch.tensor = fake
    try:
        yield
    finally:
        torch.tensor = real


class GraphedDenoiser:
    def __init__(self, policy):
        self.policy = policy
        self.m = policy.model
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.graph = None
        self.ok = True
        self.n_replays = 0
        self.trt = None        # optional TensorRT image encoder (trt_vision.TRTImageEncoder): all 3 images in one call
        self.trt_lm = None     # optional TensorRT Gemma prefix (trt_prefix.TRTPrefix): the 18-layer pass that fills the KV cache
        self.trt_expert = None  # optional TensorRT action expert (trt_expert.TRTExpert): one engine call per denoising step

    def embed_prefix(self, images, img_masks, tokens, masks):
        """PI05FlowMatching.embed_prefix with the image loop replaced by one batched TensorRT call."""
        m = self.m
        if self.trt is None or len(images) != self.trt.batch:
            return m.embed_prefix(images, img_masks, tokens, masks)
        feats = self.trt(images)
        embs, pad_masks, att = [], [], []
        for f, img_mask in zip(feats, img_masks):
            f = f.to(tokens.device)
            bsize, n = f.shape[:2]
            embs.append(f); pad_masks.append(img_mask[:, None].expand(bsize, n)); att += [0] * n
        lang = m.paligemma_with_expert.embed_language_tokens(tokens)
        lang = lang * math.sqrt(lang.shape[-1])
        embs.append(lang.to(embs[0].dtype)); pad_masks.append(masks); att += [0] * lang.shape[1]
        embs = torch.cat(embs, dim=1); pad_masks = torch.cat(pad_masks, dim=1)
        att = torch.tensor(att, dtype=torch.bool, device=pad_masks.device)[None, :].expand(pad_masks.shape[0], -1)
        return embs, pad_masks, att

    # ---- static buffers + capture ----
    def _capture(self, prefix_pad_masks, pkv, x_t):
        m = self.m
        self.s_ppm = prefix_pad_masks.clone()
        self.s_cache = DynamicCache()
        for i, layer in enumerate(pkv.layers):
            self.s_cache.update(layer.keys.clone(), layer.values.clone(), i)
        self.s_x = x_t.clone()
        self.s_t = torch.ones(x_t.shape[0], dtype=torch.float32, device=x_t.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with _const_tensors():   # one context: the warm-up fills the constant cache the capture then reads
            with torch.cuda.stream(side), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):   # warm-up (allocator, autotune) before capture
                    m.denoise_step(prefix_pad_masks=self.s_ppm, past_key_values=self.s_cache, x_t=self.s_x, timestep=self.s_t)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.s_out = m.denoise_step(prefix_pad_masks=self.s_ppm, past_key_values=self.s_cache, x_t=self.s_x, timestep=self.s_t)
        self.shape = (tuple(prefix_pad_masks.shape), tuple(x_t.shape), tuple(pkv.layers[0].keys.shape))

    def _load_cache(self, prefix_pad_masks, pkv):
        self.s_ppm.copy_(prefix_pad_masks)
        for i, layer in enumerate(pkv.layers):
            self.s_cache.layers[i].keys.copy_(layer.keys)
            self.s_cache.layers[i].values.copy_(layer.values)

    def _step(self, x_t, t):
        self.s_x.copy_(x_t)
        self.s_t.fill_(float(t))
        self.graph.replay()
        self.n_replays += 1
        return self.s_out.clone()

    # ---- drop-in for PI05FlowMatching.sample_actions ----
    @torch.no_grad()
    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
        m = self.m
        if num_steps is None:
            num_steps = m.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = m.sample_noise((bsize, m.config.chunk_size, m.config.max_action_dim), device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = m._prepare_attention_masks_4d(prefix_att_2d_masks)
        use_trt_expert = self.trt_expert is not None and bsize == 1 and prefix_embs.shape[1] == self.trt_expert.L
        if self.trt_lm is not None and bsize == 1 and prefix_embs.shape[1] == self.trt_lm.L:
            # with a TensorRT expert of the same dtype the KV cache stays in the prefix engine's buffers (no PyTorch copy)
            direct = use_trt_expert and self.trt_lm.dt == self.trt_expert.dt
            past_key_values = self.trt_lm(prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids, as_cache=not direct)
            if use_trt_expert:
                self.trt_expert.load(prefix_pad_masks, past_key_values, self.trt_lm)
        else:
            m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
            _, past_key_values = m.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids, past_key_values=None,
                inputs_embeds=[prefix_embs, None], use_cache=True)
            if use_trt_expert:
                self.trt_expert.load(prefix_pad_masks, past_key_values)

        x_t = noise
        use_graph = self.ok and bsize == 1 and not use_trt_expert
        if use_graph:
            try:
                shape = (tuple(prefix_pad_masks.shape), tuple(x_t.shape), tuple(past_key_values.layers[0].keys.shape))
                if self.graph is None or shape != self.shape:
                    self._capture(prefix_pad_masks, past_key_values, x_t)
                self._load_cache(prefix_pad_masks, past_key_values)
            except Exception as e:   # never fail an inference over the fast path
                print(f"[fast_denoise] capture failed, eager fallback: {e!r}", flush=True)
                self.ok = False
                use_graph = False

        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = 1.0 + step * dt
            if use_trt_expert:
                def denoise_step_partial_call(input_x_t, _t=time):
                    return self.trt_expert.step(input_x_t, _t)
            elif use_graph:
                def denoise_step_partial_call(input_x_t, _t=time):
                    return self._step(input_x_t, _t)
            else:
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

                def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                    return m.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                                          x_t=input_x_t, timestep=current_timestep)

            if m._rtc_enabled():
                v_t = m.rtc_processor.denoise_step(
                    x_t=x_t, prev_chunk_left_over=kwargs.get("prev_chunk_left_over"), inference_delay=kwargs.get("inference_delay"),
                    time=time, original_denoise_step_partial=denoise_step_partial_call, execution_horizon=kwargs.get("execution_horizon"))
            else:
                v_t = denoise_step_partial_call(x_t)
            x_t = x_t + dt * v_t
        return x_t


def install(policy) -> GraphedDenoiser:
    """Route policy.model.sample_actions through the graphed denoiser."""
    gd = GraphedDenoiser(policy)
    policy.model.sample_actions = gd.sample_actions
    return gd
