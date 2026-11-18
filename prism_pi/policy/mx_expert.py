"""Custom executor for the pi0.5 ACTION EXPERT denoising step (policy server only).

One inference runs the 300M action expert 10 times over 50 action tokens. That workload is not bound by arithmetic
(27 GFLOP per step) but by (a) streaming every expert weight once per step and (b) the number of small operations:
the bf16 PyTorch step is ~4.4 ms as a CUDA-graph replay, a TensorRT fp16 engine ~2.4 ms. Here one step is 9 kernels
per layer (mxe.cu, compiled with nvcc, loaded through the CUDA driver API like mx_prefix.py):
  adaptive RMSNorm + quantize  ->  ONE linear for q|k|v
  ->  attention (one fp16 tensor-core kernel: RoPE, q k^T, softmax and p v per head, split over key ranges)
  ->  merge the key ranges + quantize  ->  o linear + gated residual  ->  adaptive RMSNorm + quantize
  ->  ONE linear for gate|up  ->  gelu(gate) * up + quantize  ->  down linear + gated residual
The linears are hand-written tensor-core kernels on FP8 E4M3 operands with fp16 accumulation (the FP8 instruction that
runs at twice the fp16 rate on sm_120, which no library ships): 297 MB of weights per step instead of 623 MB. Scales are
powers of two, one per weight row and one per token (computed from the data at every call: no calibration).
Everything that depends only on the time t (time embedding, time MLP, the scale / shift / gate of all 37 adaptive
norms) is computed once per distinct t and cached; each t gets its own CUDA graph of the whole step.

Interface = trt_expert.TRTExpert: load(prefix_pad_masks, past_key_values, trt_prefix) once per inference, then
step(x_t (1, S, A) fp32, t) -> v_t (1, S, A) fp32 per denoising step (RTC guidance acts between steps, untouched).
Measured 2026-11-16 on the RTX 5090 laptop: per-step velocity within ~1% of the PyTorch bf16 expert.
"""
import ctypes, math, time
import torch
import torch.nn.functional as F
from prism_pi.policy.mx_prefix import _Kernels, supported   # noqa: F401  (same cubin loader)

E8 = torch.float8_e4m3fn
NULL = ctypes.c_void_p(0)
QT = 16.0
NTP = 12             # key tiles (8 keys) per attention part: must match mxe.cu


def q_rows(w):
    """(N, K) fp32 -> E4M3 bytes in fragment order, power-of-two scale per row (fp32) with the row max mapped to <= QT."""
    e = torch.ceil(torch.log2(w.abs().amax(-1, keepdim=True).clamp_min(1e-30) / QT)); s = torch.exp2(e)
    return w_frag((w / s).to(E8)), s.squeeze(-1).float().contiguous()


def w_frag(wq):
    """(N, K) E4M3 bytes -> tensor-core fragment order [row tile N/16][chunk K/32][reg 4][lane 32][byte 4] (see mxe.cu):
    reg = (k % 32) / 16 * 2 + (row % 16) / 8, lane = (row % 8) * 4 + (k % 16) / 4."""
    N, K = wq.shape
    b = wq.contiguous().view(torch.uint8).view(N // 16, 2, 8, K // 32, 2, 4, 4)          # [tile][hi][g][chunk][half][t][byte]
    return b.permute(0, 3, 4, 1, 2, 5, 6).contiguous().view(-1)


class MXExpert:
    def __init__(self, m, L=864, log=print, graph=True):
        ex = m.paligemma_with_expert.gemma_expert.model; cfg = ex.config
        self.m = m; self.L, self.T, self.A = L, m.config.chunk_size, m.config.max_action_dim
        self.D, self.H, self.hd, self.I, self.n_layers = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim, cfg.intermediate_size, cfg.num_hidden_layers
        assert cfg.num_key_value_heads == 1 and self.hd == 256 and self.H * self.hd == 2048 and L % 16 == 0 and self.T <= 56 and self.D % 1024 == 0 and (self.H * self.hd) % 1024 == 0 and self.I % 2048 == 0 and L + self.T <= 1024 and (self.H * self.hd) % 256 == 0 and self.D % 1024 == 0 and self.I % 2048 == 0
        self.eps = float(cfg.rms_norm_eps); self.scaling = float(self.hd ** -0.5); self.dt = torch.float16
        self.K = _Kernels(log, "mxe.cu", (b"adarms_quant", b"gelu_mul_quant", b"lin_fp8", b"attn", b"combine_quant"))
        f32 = lambda t: t.detach().to("cuda", torch.float32)
        t0 = time.time(); self.W = []; dw, db = [], []
        for lyr in ex.layers[: self.n_layers]:
            a, mm = lyr.self_attn, lyr.mlp
            self.W.append(dict(qkv=q_rows(torch.cat([f32(a.q_proj.weight), f32(a.k_proj.weight), f32(a.v_proj.weight)], 0)), o=q_rows(f32(a.o_proj.weight)),
                               gu=q_rows(torch.cat([f32(mm.gate_proj.weight), f32(mm.up_proj.weight)], 0)), down=q_rows(f32(mm.down_proj.weight))))
            for nrm in (lyr.input_layernorm, lyr.post_attention_layernorm):
                dw.append(f32(nrm.dense.weight)); db.append(f32(nrm.dense.bias))
        dw.append(f32(ex.norm.dense.weight)); db.append(f32(ex.norm.dense.bias))
        self.dense_w = torch.stack(dw); self.dense_b = torch.stack(db)                    # (2 n + 1, 3 D, cond), (2 n + 1, 3 D)
        self.tw = [(f32(l.weight), f32(l.bias)) for l in (m.time_mlp_in, m.time_mlp_out)]
        self.win_t = f32(m.action_in_proj.weight).t().contiguous(); self.b_in = f32(m.action_in_proj.bias)
        self.wout_t = f32(m.action_out_proj.weight).t().contiguous(); self.b_out = f32(m.action_out_proj.bias)
        self.inv_freq = f32(ex.rotary_emb.inv_freq)
        self.Wt, self.min_period, self.max_period = m.action_in_proj.out_features, m.config.min_period, m.config.max_period
        z = lambda *s, dt=torch.float16: torch.zeros(*s, dtype=dt, device="cuda")
        T, D, I, H, hd, nn_ = self.T, self.D, self.I, self.H, self.hd, 2 * self.n_layers + 1
        self.h = z(T, D); self.qkv = z(T, (H + 2) * hd); self.gu = z(T, 2 * I); self.fout = z(T, D, dt=torch.float32)
        self.xq = z(56 * D, dt=torch.uint8); self.sx = z(56, dt=torch.float32); self.oq = z(56 * H * hd, dt=torch.uint8); self.so = z(56, dt=torch.float32)   # fragment order
        self.gq = z(56 * I, dt=torch.uint8); self.sg = z(56, dt=torch.float32)
        # attention (mxe.cu): the prefix cache of every layer in operand order, cut into parts of 8 NTP keys; one more part = the suffix
        assert L % (8 * NTP) == 0 and T <= 8 * NTP, "prefix length must be a multiple of the attention part"
        self.P = L // (8 * NTP) + 1; self.MT = -(-T // 16)
        assert self.P <= 16 and (H * self.MT) % 8 == 0
        self.kf = z(self.n_layers, L * hd); self.vf = z(self.n_layers, L * hd)
        self.km = z(self.P * NTP * 8, dt=torch.float32)                                          # additive key mask
        self.po = z(self.P * H * self.MT * 16 * hd); self.st = z(self.P * H * self.MT * 16 * 2, dt=torch.float32)
        self.cs = z(2, T, hd // 2, dt=torch.float32)
        self.s_x = z(T, self.A, dt=torch.float32)
        self.out = z(1, T, self.A, dt=torch.float32)
        self.s_prev = z(T, self.A, dt=torch.float32); self.s_w = z(T, 1, dt=torch.float32)   # RTC: previous chunk's leftover, prefix weights
        self._loops = {}
        self._mods = {}; self._graphs = {}; self.use_graph = graph; self._arange = torch.arange(T, device="cuda"); self.n_steps = 0; self.graph = None; self.shared_kv = False
        nb = sum(w[0].numel() for d in self.W for w in d.values())
        log(f"[mx_expert] {self.n_layers} expert layers quantized to FP8 in {time.time() - t0:.1f} s ({nb / 2**20:.0f} MB), L={L}, S={T}")

    def _modulation(self, t: float):
        mo = self._mods.get(t)
        if mo is None:   # depends on t only: time embedding -> time MLP -> the scale / shift / gate of every adaptive norm
            from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding
            with torch.autocast("cuda", enabled=False):
                tt = torch.tensor([t], dtype=torch.float32, device="cuda")
                te = create_sinusoidal_pos_embedding(tt, self.Wt, min_period=self.min_period, max_period=self.max_period, device=tt.device).to(torch.float32)
                cond = F.silu(F.linear(F.silu(F.linear(te, *self.tw[0])), *self.tw[1]))[0]             # (cond,)
                mod = (self.dense_w @ cond + self.dense_b).view(-1, 3, self.D)
                mo = (torch.stack((1.0 + mod[:, 0], mod[:, 1]), 1).contiguous(), mod[:, 2].contiguous())
            self._mods[t] = mo
        return mo

    def _forward(self, s_mod, s_gate):
        with torch.autocast("cuda", enabled=False):
            K, T, D, I, L, H, hd = self.K, self.T, self.D, self.I, self.L, self.H, self.hd
            self.h.copy_(torch.addmm(self.b_in, self.s_x, self.win_t))
            for i, w in enumerate(self.W):
                K.launch(b"adarms_quant", T, 256, self.h, s_mod[2 * i], self.xq, self.sx, NULL, D, self.eps)
                K.launch(b"lin_fp8", (H + 2) * hd // 16, 32 * 8, self.xq, w["qkv"][0], w["qkv"][1], self.sx, self.qkv, NULL, T, (H + 2) * hd, D, 0)
                K.launch(b"attn", self.P * (H * self.MT // 8), 256, self.qkv, self.cs, self.kf[i], self.vf[i], self.km, self.po, self.st, T, H, self.MT, L, self.scaling, smem=NTP * hd * 16)
                K.launch(b"combine_quant", T, 256, self.po, self.st, self.oq, self.so, H, self.MT, self.P)
                K.launch(b"lin_fp8", D // 16, 32 * 8, self.oq, w["o"][0], w["o"][1], self.so, self.h, s_gate[2 * i], T, D, H * hd, 1)
                K.launch(b"adarms_quant", T, 256, self.h, s_mod[2 * i + 1], self.xq, self.sx, NULL, D, self.eps)
                K.launch(b"lin_fp8", 2 * I // 16, 32 * 8, self.xq, w["gu"][0], w["gu"][1], self.sx, self.gu, NULL, T, 2 * I, D, 0)
                K.launch(b"gelu_mul_quant", T, 512, self.gu, self.gq, self.sg, I)
                K.launch(b"lin_fp8", D // 16, 32 * 16, self.gq, w["down"][0], w["down"][1], self.sg, self.h, s_gate[2 * i + 1], T, D, I, 1)
            K.launch(b"adarms_quant", T, 256, self.h, s_mod[2 * self.n_layers], NULL, NULL, self.fout, D, self.eps)
            self.out[0].copy_(torch.addmm(self.b_out, self.fout, self.wout_t))

    def _capture(self, t: float):
        mod, gate = self._modulation(t)
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._forward(mod, gate)
        torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._forward(mod, gate)
        if len(self._graphs) >= 64:          # unusual schedules (many distinct t): do not grow without bound
            self._graphs.clear()
        self._graphs[t] = g
        return g

    @torch.no_grad()
    def load(self, prefix_pad_masks, past_key_values=None, trt_prefix=None):
        """Once per inference: the prefix KV cache (from a prefix backend's fp16 buffers or a DynamicCache), validity mask, positions."""
        L, n, hd = self.L, self.n_layers, self.hd
        if trt_prefix is not None and trt_prefix.keys.shape[0] == n and trt_prefix.keys.shape[-2] == L:
            keys, vals = trt_prefix.keys[:, 0, 0], trt_prefix.values[:, 0, 0]; self.shared_kv = True
        else:
            keys = torch.stack([l.keys[0, 0] for l in past_key_values.layers]); vals = torch.stack([l.values[0, 0] for l in past_key_values.layers]); self.shared_kv = False
        # (n, L, hd) -> operand order (kloc / vloc in mxe.cu)
        self.kf.view(n, L // 8, hd // 16, 8, 4, 2, 2).copy_(keys.reshape(n, L // 8, 8, hd // 16, 2, 4, 2).permute(0, 1, 3, 2, 5, 4, 6))
        self.vf.view(n, L // 16, hd // 8, 8, 4, 2, 2).copy_(vals.reshape(n, L // 16, 2, 4, 2, hd // 8, 8).permute(0, 1, 5, 6, 3, 2, 4))
        valid = prefix_pad_masks[0].to(torch.bool)
        self.km.fill_(-3.0e38); self.km[:L].copy_(torch.where(valid, 0.0, -3.0e38)); self.km[L:L + self.T].zero_()
        fr = (valid.sum() + self._arange).float()[:, None] * self.inv_freq[None, :]
        self.cs[0].copy_(fr.cos()); self.cs[1].copy_(fr.sin())

    @torch.no_grad()
    def step(self, x_t, t: float):
        t = float(t)
        self.s_x.copy_(x_t[0])
        if self.use_graph:
            (self._graphs.get(t) or self._capture(t)).replay()
        else:
            self._forward(*self._modulation(t))
        self.n_steps += 1
        return self.out.clone()

    # ---- the whole denoising loop as ONE CUDA graph (no Python between the steps) ----
    @staticmethod
    def _guidance_weight(time: float, max_gw: float) -> float:
        """RTCProcessor.denoise_step's guidance weight for this time (same expressions, same clamping)."""
        tau = torch.as_tensor(1 - time); mg = torch.as_tensor(max_gw)
        sq = (1 - tau) ** 2
        inv_r2 = (sq + tau ** 2) / sq
        c = torch.nan_to_num((1 - tau) / tau, posinf=mg)
        return float(torch.minimum(torch.nan_to_num(c * inv_r2, posinf=mg), mg))

    def _loop_body(self, num_steps: int, max_gw: float):
        dt = -1.0 / num_steps
        x, v, P, Wt = self.s_x, self.out[0], self.s_prev, self.s_w
        for s in range(num_steps):
            time_ = 1.0 + s * dt
            self._forward(*self._modulation(float(time_)))
            # RTC prefix guidance in closed form. LeRobot computes correction = d(x1)/d(x_t) applied to err with
            # x1 = x_t - time * v_t and v_t NOT a function of x_t in the graph, i.e. correction = err = (prev - x1) * w.
            # With w = 0 (no leftover chunk) this is the plain Euler step x += dt * v.
            gw = self._guidance_weight(time_, max_gw)
            x.add_((v - gw * ((P - (x - time_ * v)) * Wt)) * dt)

    @torch.no_grad()
    def sample(self, noise, num_steps: int, prev_chunk_left_over=None, weights=None, max_guidance_weight: float = 0.0):
        """noise (1, S, A) -> x after num_steps Euler steps. prev_chunk_left_over (S, A) zero-padded and weights (S,)
        = RTC prefix guidance (None: plain denoising). load() must have been called for this observation."""
        self.s_x.copy_(noise[0])
        if prev_chunk_left_over is None:
            self.s_w.zero_()
        else:
            self.s_prev.copy_(prev_chunk_left_over); self.s_w.copy_(weights.reshape(-1, 1))
        key = (int(num_steps), float(max_guidance_weight))
        g = self._loops.get(key)
        if g is None:
            with torch.autocast("cuda", enabled=False):
                side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
                keep = self.s_x.clone()
                with torch.cuda.stream(side):
                    self._loop_body(*key)                  # warm-up (also fills the per-t modulation cache outside capture)
                torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
                self.s_x.copy_(keep)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._loop_body(*key)
                self.s_x.copy_(keep)
            if len(self._loops) >= 8:
                self._loops.clear()
            self._loops[key] = g
        g.replay()
        self.n_steps += int(num_steps)
        return self.s_x.clone()[None]
