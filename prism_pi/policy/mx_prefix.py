"""Custom executor for the pi0.5 Gemma PREFIX pass with MXFP8 linears (policy server only).

Why: on consumer Blackwell (sm_120) TensorRT 11.3 maps FP8 Q/DQ onto an Ada-generation GEMM kernel with fp32
accumulation, which runs at the fp16 rate. The full-rate FP8 path on this GPU is the block-scaled ("MX") tensor-core
instruction, which cuBLASLt exposes (torch.nn.functional.scaled_mm): FP8 E4M3 data with one power-of-two E8M0 scale
per 32 elements along the contraction dimension, for BOTH operands. Measured on the RTX 5090 laptop (175 W cap):
864 x 2048 x 16384 GEMM 0.49 ms as TensorRT fp16 -> 0.25 ms as MXFP8.

What runs here, per Gemma layer (L = 864 tokens, batch 1), everything on torch's current stream:
  rmsnorm + quantize (1 kernel)  ->  ONE GEMM for q|k|v  ->  RoPE + split (1 kernel; k, v land in the KV buffers)
  ->  q k^T (fp16 GEMM)  ->  mask + softmax (1 kernel, fp32 inside)  ->  p v (fp16 GEMM)  ->  quantize  ->  o GEMM
  ->  rmsnorm + quantize  ->  ONE GEMM for gate|up  ->  gelu(gate) * up + quantize (1 kernel, fp32 inside)  ->  down GEMM
The kernels are in mxq.cu (compiled once with nvcc to a cubin, loaded through the CUDA driver API). The last layer
stops after its k / v (nothing downstream of the last KV is used). The whole pass is replayed as one CUDA graph.

Weights are quantized once at construction (from whatever dtype the model holds). Activations are quantized at every
call with scales computed from the data, so there is no calibration set.
Interface = trt_prefix.TRTPrefix: __call__(inputs_embeds (1,L,D), attention_mask_4d (1,1,L,L) additive, position_ids (1,L),
out_dtype, as_cache) -> DynamicCache | None, results in .keys / .values (layers, 1, kv_heads, L, head_dim) fp16.
"""
import ctypes, hashlib, os, subprocess, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType, SwizzleType

from prism_pi.paths import TRT_CACHE as CACHE

ROOT = Path(__file__).resolve().parent   # the kernel sources (mxq.cu, mxe.cu) sit next to this module
E8, E8M0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
BW, SWZ = ScalingType.BlockWise1x32, SwizzleType.SWIZZLE_32_4_4
MASK_MIN = -6.0e4
KERNELS = (b"mx_quant", b"mx_quant_gelu_mul", b"mx_rmsnorm_quant", b"rope_split", b"attn_softmax")


def supported() -> bool:
    """Block-scaled FP8 tensor-core GEMMs need Blackwell (compute capability 10.0+; measured on 12.0)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10 and hasattr(F, "scaled_mm")


def to_blocked(s):
    """(rows, K/32) scale matrix -> cuBLASLt's 32_4_4 swizzled layout (rows padded to 128, columns to 4)."""
    r, c = s.shape; rb, cb = -(-r // 128), -(-c // 4)
    p = torch.zeros(rb * 128, cb * 4, dtype=s.dtype, device=s.device); p[:r, :c] = s
    return p.view(rb, 128, cb, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def scale_len(rows: int, K: int) -> int:
    return (-(-rows // 128)) * (-(-(K // 32) // 4)) * 512


def mx_quant(x):
    """Reference quantizer (weights): x (rows, K) -> E4M3 data (rows, K), power-of-two scales (rows, K/32) fp32."""
    r, k = x.shape; xb = x.float().view(r, k // 32, 32)
    e = torch.ceil(torch.log2(xb.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0)).clamp(-127, 127)
    s = torch.exp2(e)
    return (xb / s).view(r, k).to(E8), s.squeeze(-1)


class _Kernels:
    """mxq.cu -> cubin (nvcc, cached by source hash + GPU arch) -> CUDA driver module in torch's context."""

    def __init__(self, log=print, source: str = "mxq.cu", names=KERNELS):
        src = (ROOT / source).read_bytes()
        cc = "sm_%d%d" % torch.cuda.get_device_capability(0)
        CACHE.mkdir(exist_ok=True)
        cubin = CACHE / f"{Path(source).stem}_{hashlib.sha1(src).hexdigest()[:10]}_{cc}.cubin"
        if not cubin.exists():
            nvcc = os.environ.get("YAM_NVCC") or next((p for p in ("/usr/local/cuda/bin/nvcc", "/usr/local/cuda-13.3/bin/nvcc", "nvcc")
                                                       if p == "nvcc" or os.path.exists(p)))
            t0 = time.time()
            subprocess.run([nvcc, "-w", "-O3", f"-arch={cc}", "-cubin", "-o", str(cubin), str(ROOT / source)], check=True)
            log(f"[mx_prefix] kernels compiled in {time.time() - t0:.0f} s -> {cubin.name}")
        torch.zeros(1, device="cuda")                              # torch's context is current on this thread
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.mod = ctypes.c_void_p()
        rc = self.cu.cuModuleLoadData(ctypes.byref(self.mod), cubin.read_bytes())
        if rc:
            raise RuntimeError(f"cuModuleLoadData failed ({rc})")
        self.fn = {}; self._smem_ok = set()
        for n in names:
            f = ctypes.c_void_p()
            if self.cu.cuModuleGetFunction(ctypes.byref(f), self.mod, n):
                raise RuntimeError(f"kernel {n!r} not found in {cubin.name}")
            self.fn[n] = f

    def launch(self, name, grid: int, block: int, *args, smem: int = 0):
        """args: torch tensor or int > 2^31 -> void*, int -> int32, float -> float32, ctypes values as they are.
        smem = bytes of dynamic shared memory (`extern __shared__`); more than 48 KB needs the per-function opt-in."""
        if smem > 48 * 1024 and (name, smem) not in self._smem_ok:
            if self.cu.cuFuncSetAttribute(self.fn[name], 8, ctypes.c_int(smem)):      # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                raise RuntimeError(f"kernel {name!r}: {smem} bytes of shared memory not available")
            self._smem_ok.add((name, smem))
        vals = [a if isinstance(a, ctypes._SimpleCData) else ctypes.c_float(a) if isinstance(a, float) else
                ctypes.c_void_p(a.data_ptr()) if torch.is_tensor(a) else (ctypes.c_void_p(a) if a > 0x7FFFFFFF else ctypes.c_int(a)) for a in args]
        arr = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        rc = self.cu.cuLaunchKernel(self.fn[name], ctypes.c_uint(grid), 1, 1, ctypes.c_uint(block), 1, 1, ctypes.c_uint(smem),
                                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), arr, None)
        if rc:
            raise RuntimeError(f"cuLaunchKernel {name!r} failed ({rc})")


class _W:
    """Quantized weight of one (merged) linear, as the column-major (K, N) operand cuBLASLt wants on sm_120:
    nn.Linear's (out, in) row-major storage, quantized along `in`, viewed transposed."""

    def __init__(self, *weights):
        w = torch.cat([x.detach().to("cuda", torch.float32) for x in weights], 0)
        wq, ws = mx_quant(w)
        self.b = wq.t(); self.sb = to_blocked(ws).to(E8M0); self.out_f, self.in_f = w.shape

    def __call__(self, xq, sa):
        return F.scaled_mm(xq, self.b, sa, BW, self.sb, BW, swizzle_a=SWZ, swizzle_b=SWZ, output_dtype=torch.float16)


class MXPrefix:
    def __init__(self, lm, L: int = 864, log=print, graph: bool = True):
        if not supported():
            raise RuntimeError("MXFP8 prefix needs a Blackwell GPU (compute capability >= 10.0) and torch scaled_mm")
        cfg = lm.config
        self.L, self.D, self.H, self.hd = L, cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
        self.I = cfg.intermediate_size; self.n_layers = cfg.num_hidden_layers; self.eps = float(cfg.rms_norm_eps)
        assert cfg.num_key_value_heads == 1 and self.D % 32 == 0 and self.I % 32 == 0 and self.D // 32 <= 1024
        self.scaling = float(self.hd ** -0.5)
        self.dt = torch.float16
        self.kern = _Kernels(log)
        self.inv_freq = lm.rotary_emb.inv_freq.detach().to("cuda", torch.float32)
        t0 = time.time()
        self.layers = []
        for i, lyr in enumerate(lm.layers[: self.n_layers]):
            a, m = lyr.self_attn, lyr.mlp
            d = dict(g1=(1.0 + lyr.input_layernorm.weight.detach().float()).to("cuda").contiguous(), qkv=_W(a.q_proj.weight, a.k_proj.weight, a.v_proj.weight))
            if i < self.n_layers - 1:
                d.update(g2=(1.0 + lyr.post_attention_layernorm.weight.detach().float()).to("cuda").contiguous(), o=_W(a.o_proj.weight),
                         gu=_W(m.gate_proj.weight, m.up_proj.weight), down=_W(m.down_proj.weight))
            self.layers.append(d)
        z = lambda *shape, dt=torch.float16: torch.zeros(*shape, dtype=dt, device="cuda")
        self.keys = z(self.n_layers, 1, 1, L, self.hd); self.values = z(self.n_layers, 1, 1, L, self.hd)
        self.qD = z(L, self.D, dt=E8); self.sD = z(scale_len(L, self.D), dt=E8M0)
        self.qI = z(L, self.I, dt=E8); self.sI = z(scale_len(L, self.I), dt=E8M0)
        self.qrot = z(self.H * L, self.hd)
        self.s_x = z(1, L, self.D); self.s_mask = z(1, 1, L, L, dt=torch.float32); self.s_pos = torch.arange(L, device="cuda")[None]
        self.graph = None; self.n_calls = 0
        nbytes = sum(w.b.numel() + w.sb.numel() for d in self.layers for w in d.values() if isinstance(w, _W))
        log(f"[mx_prefix] {self.n_layers} Gemma layers quantized to MXFP8 in {time.time() - t0:.1f} s ({nbytes / 2**30:.2f} GiB), L={L}")
        if graph:
            self._capture()

    @torch.no_grad()
    def _forward(self):
        """Over the static buffers s_x / s_mask (already clamped) / s_pos; fills keys / values."""
        with torch.autocast("cuda", enabled=False):      # the kernels read fp16 bits: no autocast re-typing of the GEMMs
            self._layers()

    def _layers(self):
        L, D, H, hd, I, K = self.L, self.D, self.H, self.hd, self.I, self.kern
        ptr = lambda t: t.data_ptr()
        h = self.s_x[0]
        mask = self.s_mask[0, 0]
        fr = self.s_pos[0].float()[:, None] * self.inv_freq[None, :]
        cs = torch.stack((fr.cos(), fr.sin())).contiguous()                                  # (2, L, hd / 2) fp32
        nb = D // 32
        for i, ly in enumerate(self.layers):
            K.launch(b"mx_rmsnorm_quant", L, nb, ptr(h), ptr(ly["g1"]), ptr(self.qD), ptr(self.sD), L, D, self.eps)
            qkv = ly["qkv"](self.qD, self.sD)                                                # (L, (H + 2) * hd)
            k, v = self.keys[i, 0, 0], self.values[i, 0, 0]
            n = L * (H + 2) * (hd // 2)
            K.launch(b"rope_split", (n + 255) // 256, 256, ptr(qkv), ptr(cs), ptr(self.qrot), ptr(k), ptr(v), L, H, hd, self.scaling)
            if i == self.n_layers - 1:
                break
            att = torch.mm(self.qrot, k.t())                                                 # (H * L, L): the single kv head serves every query head
            K.launch(b"attn_softmax", H * L, 256, ptr(att), ptr(mask), H * L, L)
            o = torch.mm(att, v).view(H, L, hd).transpose(0, 1).reshape(L, H * hd)
            K.launch(b"mx_quant", (L * nb + 127) // 128, 128, ptr(o), ptr(self.qD), ptr(self.sD), L, D, D)
            h = h + ly["o"](self.qD, self.sD)
            K.launch(b"mx_rmsnorm_quant", L, nb, ptr(h), ptr(ly["g2"]), ptr(self.qD), ptr(self.sD), L, D, self.eps)
            gu = ly["gu"](self.qD, self.sD)                                                  # (L, 2 * I): gate | up
            K.launch(b"mx_quant_gelu_mul", (L * (I // 32) + 127) // 128, 128, ptr(gu), ptr(gu) + 2 * I, ptr(self.qI), ptr(self.sI), L, I, 2 * I)
            h = h + ly["down"](self.qI, self.sI)

    def _capture(self):
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._forward()
        self.graph = g

    @torch.no_grad()
    def __call__(self, inputs_embeds, attention_mask_4d, position_ids, out_dtype=torch.bfloat16, as_cache=True):
        assert inputs_embeds.shape[1] == self.L, f"executor built for L={self.L}, got {inputs_embeds.shape[1]}"
        self.s_x.copy_(inputs_embeds[0:1]); self.s_mask.copy_(attention_mask_4d[0:1]).clamp_(min=MASK_MIN); self.s_pos.copy_(position_ids[0:1])
        if self.graph is not None:
            self.graph.replay()
        else:
            self._forward()
        self.n_calls += 1
        if not as_cache:
            return None
        from transformers.cache_utils import DynamicCache
        c = DynamicCache()
        for i in range(self.n_layers):
            c.update(self.keys[i].to(out_dtype), self.values[i].to(out_dtype), i)
        return c
