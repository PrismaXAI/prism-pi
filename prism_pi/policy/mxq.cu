// MXFP8 quantization kernels for the Gemma prefix executor (mx_prefix.py).
// One thread per 32-element block along the contraction dimension: block max -> power-of-two E8M0 scale -> E4M3 bytes.
// Scales are written directly in cuBLASLt's 32_4_4 swizzled layout. Loaded through the CUDA driver API (cubin).
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint8_t to_e4m3(float v) {          // round-to-nearest-even, saturating at +-448
    uint32_t sign = (__float_as_uint(v) >> 24) & 0x80u;
    float a = fabsf(v);
    if (!(a < 448.0f)) return (uint8_t)(sign | 0x7Eu);            // also catches NaN/inf
    if (a < 0.015625f) return (uint8_t)(sign | (uint32_t)rintf(a * 512.0f));   // subnormals (and the first normal, 8)
    int p = (int)((__float_as_uint(a) >> 23) & 0xFF) - 127;       // a = m * 2^p, m in [1, 2)
    float m = __uint_as_float((__float_as_uint(a) & 0x007FFFFFu) | 0x3F800000u);
    uint32_t mant = (uint32_t)rintf((m - 1.0f) * 8.0f);          // 0..8; 8 carries into the exponent
    return (uint8_t)(sign | ((((uint32_t)(p + 7)) << 3) + mant));
}

__device__ __forceinline__ int block_exp(float amax) {            // smallest e with amax * 2^-e <= 448 (= 1.75 * 2^8)
    if (!(amax > 0.0f)) return -127;
    uint32_t u = __float_as_uint(amax);
    int p = (int)((u >> 23) & 0xFF) - 127;
    int e = p - 8 + (((u & 0x007FFFFFu) > 0x00600000u) ? 1 : 0);  // mantissa > 1.75
    return max(-127, min(127, e));
}

__device__ __forceinline__ void emit(const float *v, uint8_t *q, uint8_t *s, int row, int blk, int cb) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    int e = block_exp(amax);
    float inv = __uint_as_float((uint32_t)(127 - e) << 23);       // 2^-e
    uint32_t w[8];
    for (int i = 0; i < 8; i++)
        w[i] = (uint32_t)to_e4m3(v[4 * i] * inv) | ((uint32_t)to_e4m3(v[4 * i + 1] * inv) << 8) |
               ((uint32_t)to_e4m3(v[4 * i + 2] * inv) << 16) | ((uint32_t)to_e4m3(v[4 * i + 3] * inv) << 24);
    uint32_t *qo = (uint32_t *)q;
    for (int i = 0; i < 8; i++) qo[i] = w[i];
    int tile = (row >> 7) * cb + (blk >> 2);
    s[tile * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (blk & 3)] = (uint8_t)(e + 127);
}

extern "C" __global__ void mx_quant(const __half *x, uint8_t *q, uint8_t *s, int rows, int K, int stride) {
    int nb = K >> 5, idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nb) return;
    int row = idx / nb, blk = idx - row * nb;
    const __half2 *p = (const __half2 *)(x + (size_t)row * stride + blk * 32);
    float v[32];
    for (int i = 0; i < 16; i++) { float2 f = __half22float2(p[i]); v[2 * i] = f.x; v[2 * i + 1] = f.y; }
    emit(v, q + (size_t)row * K + blk * 32, s, row, blk, (nb + 3) >> 2);
}

// v = gelu_tanh(g) * u, computed in fp32 (no fp16 overflow of the product), quantized in the same pass
extern "C" __global__ void mx_quant_gelu_mul(const __half *g, const __half *u, uint8_t *q, uint8_t *s, int rows, int K, int stride) {
    int nb = K >> 5, idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nb) return;
    int row = idx / nb, blk = idx - row * nb;
    size_t off = (size_t)row * K + blk * 32, offi = (size_t)row * stride + blk * 32;
    const __half2 *pg = (const __half2 *)(g + offi), *pu = (const __half2 *)(u + offi);
    float v[32];
    for (int i = 0; i < 16; i++) {
        float2 a = __half22float2(pg[i]), b = __half22float2(pu[i]);
        v[2 * i]     = 0.5f * a.x * (1.0f + tanhf(0.7978845608028654f * (a.x + 0.044715f * a.x * a.x * a.x))) * b.x;
        v[2 * i + 1] = 0.5f * a.y * (1.0f + tanhf(0.7978845608028654f * (a.y + 0.044715f * a.y * a.y * a.y))) * b.y;
    }
    emit(v, q + off, s, row, blk, (nb + 3) >> 2);
}

// RMSNorm (Gemma: x * rsqrt(mean(x^2) + eps) * (1 + w), in fp32) fused with the MXFP8 quantization of its output.
// One thread block per row, one thread per 32-element block (K / 32 <= 1024). gain = 1 + w, fp32.
extern "C" __global__ void mx_rmsnorm_quant(const __half *x, const float *gain, uint8_t *q, uint8_t *s, int rows, int K, float eps) {
    __shared__ float part[1024];
    int row = blockIdx.x, blk = threadIdx.x, nb = K >> 5;
    const __half2 *p = (const __half2 *)(x + (size_t)row * K + blk * 32);
    float v[32], acc = 0.0f;
    for (int i = 0; i < 16; i++) { float2 f = __half22float2(p[i]); v[2 * i] = f.x; v[2 * i + 1] = f.y; acc += f.x * f.x + f.y * f.y; }
    part[blk] = acc; __syncthreads();
    for (int st = 1; st < nb; st <<= 1) { if ((blk & (2 * st - 1)) == 0 && blk + st < nb) part[blk] += part[blk + st]; __syncthreads(); }
    float inv = rsqrtf(part[0] / (float)K + eps);
    const float *g = gain + blk * 32;
    for (int i = 0; i < 32; i++) v[i] = v[i] * inv * g[i];
    emit(v, q + (size_t)row * K + blk * 32, s, row, blk, (nb + 3) >> 2);
}

// Splits the merged q|k|v GEMM output (L, (H + 2) * hd), applies RoPE to q and k, scales q by the attention scaling,
// and writes q as (H, L, hd), k and v as (L, hd) (straight into the KV-cache buffers). cs = cos | sin tables (2, L, hd / 2), fp32.
extern "C" __global__ void rope_split(const __half *qkv, const float *cs, __half *qo, __half *ko, __half *vo, int L, int H, int hd, float qscale) {
    int half_ = hd >> 1, per_row = (H + 2) * half_;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * per_row) return;
    int row = idx / per_row, c = idx - row * per_row, head = c / half_, j = c - head * half_;
    const __half *src = qkv + (size_t)row * (H + 2) * hd + head * hd;
    float x1 = __half2float(src[j]), x2 = __half2float(src[j + half_]);
    if (head == H + 1) { vo[(size_t)row * hd + j] = src[j]; vo[(size_t)row * hd + j + half_] = src[j + half_]; return; }
    float co = cs[(size_t)row * half_ + j], si = cs[(size_t)(L + row) * half_ + j];
    float o1 = x1 * co - x2 * si, o2 = x2 * co + x1 * si;
    if (head == H) { ko[(size_t)row * hd + j] = __float2half(o1); ko[(size_t)row * hd + j + half_] = __float2half(o2); return; }
    __half *dst = qo + ((size_t)head * L + row) * hd;
    dst[j] = __float2half(o1 * qscale); dst[j + half_] = __float2half(o2 * qscale);
}

// In-place row softmax of the attention scores (rows = H * L, cols = L) with an additive fp32 mask (L, L) shared by the
// heads; arithmetic in fp32, storage fp16. One thread block (256 threads) per row.
extern "C" __global__ void attn_softmax(__half *att, const float *mask, int rows, int L) {
    __shared__ float red[256];
    int row = blockIdx.x, t = threadIdx.x;
    __half *a = att + (size_t)row * L; const float *m = mask + (size_t)(row % L) * L;
    float v[8], mx = -3.0e38f; int n = 0;
    for (int c = t; c < L && n < 8; c += 256, n++) { v[n] = __half2float(a[c]) + m[c]; mx = fmaxf(mx, v[n]); }
    red[t] = mx; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (t < st) red[t] = fmaxf(red[t], red[t + st]); __syncthreads(); }
    mx = red[0]; __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { v[i] = __expf(v[i] - mx); sum += v[i]; }
    red[t] = sum; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (t < st) red[t] += red[t + st]; __syncthreads(); }
    float inv = 1.0f / red[0];
    n = 0;
    for (int c = t; c < L && n < 8; c += 256, n++) a[c] = __float2half(v[n] * inv);
}
