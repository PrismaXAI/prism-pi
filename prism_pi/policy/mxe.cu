// Kernels of the custom pi0.5 ACTION EXPERT executor (mx_expert.py): one flow-matching denoising step over T (<= 56)
// suffix tokens, batch 1. Linears run on FP8 E4M3 operands with fp16 accumulation
// (mma.sync m16n8k32 f16.e4m3.e4m3.f16: on sm_120 the tensor-core instruction with twice the fp16 rate, and half the
// weight bytes to stream, which is what bounds a 50-token step). Scales are powers of two: one per weight row, one per
// token for activations (largest magnitude mapped to <= QT so fp16 partial sums stay far from overflow).
#include <cuda_fp16.h>
#include <stdint.h>
#define QT_LOG2 4            // quantization target 2^4 = 16
#define NT 7                 // token tiles of 8 (T <= 56)

__device__ __forceinline__ uint8_t to_e4m3(float v) {          // round-to-nearest-even, saturating at +-448
    uint32_t sign = (__float_as_uint(v) >> 24) & 0x80u;
    float a = fabsf(v);
    if (!(a < 448.0f)) return (uint8_t)(sign | 0x7Eu);
    if (a < 0.015625f) return (uint8_t)(sign | (uint32_t)rintf(a * 512.0f));
    int p = (int)((__float_as_uint(a) >> 23) & 0xFF) - 127;
    float m = __uint_as_float((__float_as_uint(a) & 0x007FFFFFu) | 0x3F800000u);
    return (uint8_t)(sign | ((((uint32_t)(p + 7)) << 3) + (uint32_t)rintf((m - 1.0f) * 8.0f)));
}
__device__ __forceinline__ float pow2f(int e) { return __uint_as_float((uint32_t)(127 + e) << 23); }
__device__ __forceinline__ int scale_exp(float amax) {            // smallest e with amax * 2^-e <= 2^QT_LOG2
    if (!(amax > 1e-30f)) return -60;
    if (!(amax < 1e30f)) return 60;
    uint32_t u = __float_as_uint(amax);
    int e = (int)((u >> 23) & 0xFF) - 127 - QT_LOG2 + ((u & 0x007FFFFFu) ? 1 : 0);
    return max(-60, min(60, e));
}
// Activations are stored in tensor-core FRAGMENT ORDER so that every operand load of lin_fp8 is one contiguous 128-byte
// read: byte of (token, k) -> [chunk = k / 32][tile = token / 8][reg = (k % 32) / 16][lane = (token % 8) * 4 + (k % 16) / 4][k % 4].
// Buffers hold NT * 8 = 56 token slots (slots >= T stay zero).
__device__ __forceinline__ size_t xfrag(int tok, int k) {
    return ((((size_t)(k >> 5) * NT + (tok >> 3)) * 2 + ((k >> 4) & 1)) * 32 + ((tok & 7) * 4 + ((k >> 2) & 3))) * 4 + (k & 3);
}
// block-wide reductions: shuffles inside the warp, then one step through `red` (one float per warp, <= 16 warps)
__device__ __forceinline__ float block_sum(float *red, float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.0f; for (int i = 0; i < (int)(blockDim.x >> 5); i++) r += red[i];
    __syncthreads(); return r;
}
__device__ __forceinline__ float block_max(float *red, float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0]; for (int i = 1; i < (int)(blockDim.x >> 5); i++) r = fmaxf(r, red[i]);
    __syncthreads(); return r;
}
__device__ __forceinline__ uint32_t pack4(const float *v, float q) {      // 4 consecutive values -> one word of E4M3 bytes
    return to_e4m3(v[0] * q) | (to_e4m3(v[1] * q) << 8) | (to_e4m3(v[2] * q) << 16) | (to_e4m3(v[3] * q) << 24);
}

// Adaptive RMSNorm (x * rsqrt(mean(x^2) + eps) * (1 + scale) + shift, fp32) + per-token FP8 quantization.
// mod = [1 + scale | shift] (2 x D, fp32). One block (256 threads) per token. fout != NULL: write fp32 instead (final norm).
extern "C" __global__ void adarms_quant(const __half *h, const float *mod, uint8_t *xq, float *sx, float *fout, int D, float eps) {
    __shared__ float red[16];
    int tok = blockIdx.x, t = threadIdx.x, per = D >> 8;          // per = 4 or 8 consecutive values per thread
    const __half *p = h + (size_t)tok * D + t * per; const float *m = mod + t * per;
    float v[8], m0[8], m1[8], acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) if (i < per) { v[i] = __half2float(p[i]); m0[i] = m[i]; m1[i] = m[D + i]; acc += v[i] * v[i]; }
    float inv = rsqrtf(block_sum(red, acc) / (float)D + eps), am = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) if (i < per) { v[i] = v[i] * inv * m0[i] + m1[i]; am = fmaxf(am, fabsf(v[i])); }
    if (fout) { for (int i = 0; i < per; i++) fout[(size_t)tok * D + t * per + i] = v[i]; return; }
    int e = scale_exp(block_max(red, am)); float q = pow2f(-e);
    #pragma unroll
    for (int i = 0; i < 8; i += 4) if (i < per) *(uint32_t *)(xq + xfrag(tok, t * per + i)) = pack4(v + i, q);       // xfrag: 4 consecutive k = one word
    if (t == 0) sx[tok] = pow2f(e);
}

// gelu_tanh(gate) * up (fp32) + per-token FP8 quantization. gu = (T, 2 * I): gate | up. One block (512 threads) per token.
extern "C" __global__ void gelu_mul_quant(const __half *gu, uint8_t *gq, float *sg, int I) {
    __shared__ float red[16];
    int tok = blockIdx.x, t = threadIdx.x, per = I >> 9;           // per = 4 or 8 consecutive values per thread
    const __half *pg = gu + (size_t)tok * 2 * I + t * per, *pu = pg + I;
    float v[8], u[8], am = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) if (i < per) { v[i] = __half2float(pg[i]); u[i] = __half2float(pu[i]); }
    #pragma unroll
    for (int i = 0; i < 8; i++) if (i < per) {
        float a = v[i];
        v[i] = 0.5f * a * (1.0f + tanhf(0.7978845608028654f * (a + 0.044715f * a * a * a))) * u[i];
        am = fmaxf(am, fabsf(v[i]));
    }
    int e = scale_exp(block_max(red, am)); float q = pow2f(-e);
    #pragma unroll
    for (int i = 0; i < 8; i += 4) if (i < per) *(uint32_t *)(gq + xfrag(tok, t * per + i)) = pack4(v + i, q);
    if (t == 0) sg[tok] = pow2f(e);
}

// y(T, N) = x(T, K) @ W(N, K)^T, FP8 E4M3 operands, fp16 accumulation, both operands in fragment order:
//   wf[(row tile * K / 32 + chunk) * 4 + reg][lane]   (row tile = 16 output rows; mx_expert.w_frag)
//   xf[(chunk * NT + token tile) * 2 + reg][lane]     (xfrag above)
// One block = one row tile; its KS = blockDim / 32 warps (<= 16) each take a K slice of K / 32 / KS chunks (a multiple
// of UN). Per round a warp issues the weight loads of UN chunks back to back (the step is bound by memory latency, not
// arithmetic), then the x loads, then UN * NT mma. Partial sums are fp16 per slice and added in fp32.
// mode 0: y[tok][n] = v            mode 1: y[tok][n] += v * gate[n]   (gated residual, y = hidden state)
#define RT 1
#define UN 4
extern "C" __global__ void lin_fp8(const uint32_t *xf, const uint32_t *wf, const float *sw, const float *sx, __half *y, const float *gate, int T, int N, int K, int mode) {
    __shared__ uint32_t part[16 * 32 * NT * 2];              // [K slice][lane][token tile][2 accumulator registers]
    int w = threadIdx.x >> 5, KS = blockDim.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int n0 = blockIdx.x * 16 * RT, KC = K >> 5;
    const uint32_t *wp = wf + (size_t)blockIdx.x * RT * KC * 128 + lane;
    const uint32_t *xp = xf + lane;
    uint32_t c[RT][NT][2];
    #pragma unroll
    for (int r = 0; r < RT; r++) for (int i = 0; i < NT; i++) c[r][i][0] = c[r][i][1] = 0;
    int kc = KC / KS;
    for (int ch = w * kc; ch < (w + 1) * kc; ch += UN) {
        uint32_t a[RT][UN][4], b[UN][NT][2];
        #pragma unroll
        for (int r = 0; r < RT; r++)
            #pragma unroll
            for (int u = 0; u < UN; u++) { const uint32_t *ap = wp + ((size_t)r * KC + ch + u) * 128; a[r][u][0] = ap[0]; a[r][u][1] = ap[32]; a[r][u][2] = ap[64]; a[r][u][3] = ap[96]; }
        #pragma unroll
        for (int u = 0; u < UN; u++) { const uint32_t *bp = xp + (size_t)(ch + u) * NT * 64;
            #pragma unroll
            for (int i = 0; i < NT; i++) { b[u][i][0] = bp[i * 64]; b[u][i][1] = bp[i * 64 + 32]; } }
        #pragma unroll
        for (int r = 0; r < RT; r++)
            #pragma unroll
            for (int u = 0; u < UN; u++)
                #pragma unroll
                for (int i = 0; i < NT; i++)
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                                 : "+r"(c[r][i][0]), "+r"(c[r][i][1]) : "r"(a[r][u][0]), "r"(a[r][u][1]), "r"(a[r][u][2]), "r"(a[r][u][3]), "r"(b[u][i][0]), "r"(b[u][i][1]));
    }
    if (KS > 1) {
        #pragma unroll
        for (int r = 0; r < RT; r++)
            #pragma unroll
            for (int i = 0; i < NT; i++) { uint32_t *q = part + (((size_t)(w * RT + r) * 32 + lane) * NT + i) * 2; q[0] = c[r][i][0]; q[1] = c[r][i][1]; }
        __syncthreads();
        if (w) return;
    }
    #pragma unroll
    for (int r = 0; r < RT; r++) {
        int na = n0 + r * 16 + g, nb = na + 8;
        float sa = sw[na], sb = sw[nb], ga = 1.0f, gb = 1.0f;
        if (mode) { ga = gate[na]; gb = gate[nb]; }
        #pragma unroll
        for (int i = 0; i < NT; i++) {
            float2 lo = __half22float2(*(__half2 *)&c[r][i][0]), hi = __half22float2(*(__half2 *)&c[r][i][1]);
            for (int ww = 1; ww < KS; ww++) {
                const uint32_t *q = part + (((size_t)(ww * RT + r) * 32 + lane) * NT + i) * 2;
                float2 l2 = __half22float2(*(const __half2 *)&q[0]), h2 = __half22float2(*(const __half2 *)&q[1]);
                lo.x += l2.x; lo.y += l2.y; hi.x += h2.x; hi.y += h2.y;
            }
            int tok = i * 8 + t * 2;
            if (tok < T) {
                float s = sx[tok], ya = lo.x * sa * s, yb = hi.x * sb * s;
                __half *pa = y + (size_t)tok * N + na, *pb = y + (size_t)tok * N + nb;
                if (mode) { ya = __half2float(*pa) + ya * ga; yb = __half2float(*pb) + yb * gb; }
                *pa = __float2half(ya); *pb = __float2half(yb);
            }
            if (tok + 1 < T) {
                float s = sx[tok + 1], ya = lo.y * sa * s, yb = hi.y * sb * s;
                __half *pa = y + (size_t)(tok + 1) * N + na, *pb = y + (size_t)(tok + 1) * N + nb;
                if (mode) { ya = __half2float(*pa) + ya * ga; yb = __half2float(*pb) + yb * gb; }
                *pa = __float2half(ya); *pb = __float2half(yb);
            }
        }
    }
}

// ---- attention: 2 kernels (attn -> combine_quant) --------------------------------------------------------------------
// The prefix keys / values of one layer (cache rows 0..L-1; mx_expert.load) are kept in tensor-core operand order, the
// two registers of an operand adjacent, so that the right operand of every mma is ONE 8-byte load per lane:
//   kf: element (key, dim) -> [key / 8][dim / 16][key & 7][(dim / 2) & 3][(dim / 8) & 1][dim & 1]     (right operand of q k^T)
//   vf: element (key, dim) -> [key / 16][dim / 8][dim & 7][(key / 2) & 3][(key / 8) & 1][key & 1]     (right operand of p v)
// The keys are cut into parts of NTP tiles (8 NTP keys); L is a multiple of that, and the part after the prefix holds
// this step's T suffix tokens (k with RoPE, v), which attn builds straight from the q|k|v projection.
#define AHD 256              // head dimension
#define NTP 12               // key tiles (of 8 keys) per attention part
#define MMA16(c, a0, a1, a2, a3, b0, b1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" \
        : "+r"((c)[0]), "+r"((c)[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
__device__ __forceinline__ int kloc(int key, int d) { return (((((key >> 3) * (AHD / 16) + (d >> 4)) * 8 + (key & 7)) * 4 + ((d >> 1) & 3)) * 2 + ((d >> 3) & 1)) * 2 + (d & 1); }
__device__ __forceinline__ int vloc(int key, int d) { return (((((key >> 4) * (AHD / 8) + (d >> 3)) * 8 + (d & 7)) * 4 + ((key >> 1) & 3)) * 2 + ((key >> 3) & 1)) * 2 + (key & 1); }
#define SMQ (NTP * (AHD / 16) * 32)          // uint64 words of one part of kf (= of vf)
__device__ __forceinline__ void stage(uint64_t *sm, const uint64_t *src) {     // 256 threads, every thread issues all its loads at once
    uint64_t r[SMQ / 256];
    #pragma unroll
    for (int i = 0; i < SMQ / 256; i++) r[i] = src[i * 256 + threadIdx.x];
    #pragma unroll
    for (int i = 0; i < SMQ / 256; i++) sm[i * 256 + threadIdx.x] = r[i];
}
__device__ __forceinline__ void stage_zero(uint64_t *sm) {
    #pragma unroll
    for (int i = 0; i < SMQ / 256; i++) sm[i * 256 + threadIdx.x] = 0;
    __syncthreads();
}

// softmax(q k^T) v against ONE PART of the keys for 8 (head, query tile) pairs: one block of 8 warps,
// blockIdx = part * (H * MT / 8) + group, warp w -> pair group * 8 + w = head * MT + query tile (16 query rows).
// What bounds this kernel is operand traffic and instruction count, not arithmetic: the part's keys are staged ONCE per
// block in shared memory (then the part's values, in the same buffer) and all 8 warps read them from there. Each warp
// holds its q tile in registers (RoPE and the 1 / sqrt(hd) scaling applied there), keeps its 16 x (8 NTP) scores in
// registers, takes the softmax relative to the part's own row maximum, and reuses the score registers as the left operand
// of p v. It writes the unnormalised partial output in accumulator order,
//   po[((part * H + head) * MT + query tile) * (AHD / 8) + dim tile][register][lane]   (128 contiguous bytes per store),
// and st[part][head][row] = (row maximum, sum of exp); combine_quant merges the parts.
// qkv = merged projection (T, (H + 2) * AHD); cs = cos | sin (2, T, AHD / 2) fp32;
// km = additive key mask (fp32 per key: 0, or -3e38 for prefix padding and the unused keys of the suffix part).
extern "C" __global__ void attn(const __half *qkv, const float *cs, const uint64_t *kf, const uint64_t *vf, const float *km, uint32_t *po, float *st,
                                int T, int H, int MT, int L, float qscale) {
    extern __shared__ uint64_t sm[];
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int groups = H * MT / 8, part = blockIdx.x / groups, pair = (blockIdx.x - part * groups) * 8 + w, head = pair / MT, mt = pair - head * MT, q0 = mt * 16;
    bool suffix = part * NTP * 8 >= L;
    const __half *kv = qkv + (size_t)H * AHD;                      // k | v columns of a projection row
    if (!suffix) stage(sm, kf + (size_t)part * SMQ);
    else {                                                         // this step's keys: RoPE, written in operand order
        stage_zero(sm);
        int j = threadIdx.x & (AHD / 2 - 1); __half *smh = (__half *)sm;
        __half x1[NT * 4], x2[NT * 4]; float co[NT * 4], si[NT * 4]; int r0 = threadIdx.x / (AHD / 2);       // rows r0, r0 + 2, ...: every load issued at once
        #pragma unroll
        for (int i = 0; i < NT * 4; i++) { int r = min(r0 + 2 * i, T - 1); const __half *src = kv + (size_t)r * (H + 2) * AHD;
            x1[i] = src[j]; x2[i] = src[j + AHD / 2]; co[i] = cs[(size_t)r * (AHD / 2) + j]; si[i] = cs[(size_t)(T + r) * (AHD / 2) + j]; }
        #pragma unroll
        for (int i = 0; i < NT * 4; i++) { int r = r0 + 2 * i; if (r < T) { float a1 = __half2float(x1[i]), a2 = __half2float(x2[i]);
            smh[kloc(r, j)] = __float2half(a1 * co[i] - a2 * si[i]); smh[kloc(r, j + AHD / 2)] = __float2half(a2 * co[i] + a1 * si[i]); } }
    }
    uint32_t a[AHD / 16][4];
    {   // q tile: registers 0 / 2 = row q0 + g, 1 / 3 = row q0 + g + 8; dims 16 c + 2 t (+ 8 for registers 2 / 3). RoPE pairs dim d with d + AHD / 2.
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = q0 + g + 8 * (r & 1), dofs = 8 * (r >> 1) + 2 * t; bool ok = row < T;
            const __half *qp = qkv + (size_t)(ok ? row : 0) * (H + 2) * AHD + head * AHD + dofs;
            const float *cp = cs + (size_t)(ok ? row : 0) * (AHD / 2) + dofs, *sp = cp + (size_t)T * (AHD / 2);
            uint32_t lo[AHD / 32], hi[AHD / 32]; float2 co[AHD / 32], si[AHD / 32];
            #pragma unroll
            for (int c = 0; c < AHD / 32; c++) { lo[c] = *(const uint32_t *)(qp + 16 * c); hi[c] = *(const uint32_t *)(qp + AHD / 2 + 16 * c); co[c] = *(const float2 *)(cp + 16 * c); si[c] = *(const float2 *)(sp + 16 * c); }
            #pragma unroll
            for (int c = 0; c < AHD / 32; c++) {
                float2 x1 = __half22float2(*(__half2 *)&lo[c]), x2 = __half22float2(*(__half2 *)&hi[c]);
                float s = ok ? qscale : 0.0f;
                *(__half2 *)&a[c][r] = __floats2half2_rn((x1.x * co[c].x - x2.x * si[c].x) * s, (x1.y * co[c].y - x2.y * si[c].y) * s);
                *(__half2 *)&a[c + AHD / 32][r] = __floats2half2_rn((x2.x * co[c].x + x1.x * si[c].x) * s, (x2.y * co[c].y + x1.y * si[c].y) * s);
            }
        }
    }
    __syncthreads();
    uint32_t sc[NTP][2];
    #pragma unroll
    for (int j = 0; j < NTP; j++) sc[j][0] = sc[j][1] = 0;
    #pragma unroll
    for (int c = 0; c < AHD / 16; c++) {            // chunk-major: the NTP accumulators are independent, so their mma overlap
        uint64_t b[NTP];
        #pragma unroll
        for (int j = 0; j < NTP; j++) b[j] = sm[(j * (AHD / 16) + c) * 32 + lane];
        #pragma unroll
        for (int j = 0; j < NTP; j++) MMA16(sc[j], a[c][0], a[c][1], a[c][2], a[c][3], (uint32_t)b[j], (uint32_t)(b[j] >> 32));
    }
    __syncthreads();                                // every warp is done with the keys
    if (!suffix) stage(sm, vf + (size_t)part * SMQ);
    else {
        stage_zero(sm);
        int d = threadIdx.x; __half *smh = (__half *)sm;
        __half x[NT * 8];
        #pragma unroll
        for (int r = 0; r < NT * 8; r++) x[r] = kv[(size_t)min(r, T - 1) * (H + 2) * AHD + AHD + d];
        #pragma unroll
        for (int r = 0; r < NT * 8; r++) if (r < T) smh[vloc(r, d)] = x[r];
    }
    const float2 *mp = (const float2 *)(km + part * NTP * 8 + 2 * t);
    float2 s0[NTP], s1[NTP]; float m0 = -3.0e38f, m1 = -3.0e38f;
    #pragma unroll
    for (int j = 0; j < NTP; j++) {
        float2 mk = mp[j * 4];
        s0[j] = __half22float2(*(__half2 *)&sc[j][0]); s1[j] = __half22float2(*(__half2 *)&sc[j][1]);
        s0[j].x += mk.x; s0[j].y += mk.y; s1[j].x += mk.x; s1[j].y += mk.y;
        m0 = fmaxf(m0, fmaxf(s0[j].x, s0[j].y)); m1 = fmaxf(m1, fmaxf(s1[j].x, s1[j].y));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    float e0 = 0.0f, e1 = 0.0f;
    #pragma unroll
    for (int j = 0; j < NTP; j++) {
        s0[j].x = __expf(s0[j].x - m0); s0[j].y = __expf(s0[j].y - m0); s1[j].x = __expf(s1[j].x - m1); s1[j].y = __expf(s1[j].y - m1);
        e0 += s0[j].x + s0[j].y; e1 += s1[j].x + s1[j].y;
        *(__half2 *)&sc[j][0] = __floats2half2_rn(s0[j].x, s0[j].y); *(__half2 *)&sc[j][1] = __floats2half2_rn(s1[j].x, s1[j].y);
    }
    e0 += __shfl_xor_sync(0xffffffffu, e0, 1); e0 += __shfl_xor_sync(0xffffffffu, e0, 2);
    e1 += __shfl_xor_sync(0xffffffffu, e1, 1); e1 += __shfl_xor_sync(0xffffffffu, e1, 2);
    size_t ot = (size_t)(part * H + head) * MT + mt;
    if (t == 0) { float *sp = st + (ot * 16 + g) * 2; sp[0] = m0; sp[1] = e0; sp[16] = m1; sp[17] = e1; }
    __syncthreads();
    uint32_t o[AHD / 8][2];
    #pragma unroll
    for (int j = 0; j < AHD / 8; j++) o[j][0] = o[j][1] = 0;
    #pragma unroll
    for (int c = 0; c < NTP / 2; c++) {
        uint64_t b[AHD / 8];
        #pragma unroll
        for (int j = 0; j < AHD / 8; j++) b[j] = sm[(c * (AHD / 8) + j) * 32 + lane];
        #pragma unroll
        for (int j = 0; j < AHD / 8; j++) MMA16(o[j], sc[2 * c][0], sc[2 * c][1], sc[2 * c + 1][0], sc[2 * c + 1][1], (uint32_t)b[j], (uint32_t)(b[j] >> 32));
    }
    uint32_t *op = po + ot * (AHD / 8) * 64 + lane;
    #pragma unroll
    for (int j = 0; j < AHD / 8; j++) { op[j * 64] = o[j][0]; op[j * 64 + 32] = o[j][1]; }
}

// Merge the attention parts of one token (softmax across parts: weight exp(max_p - max) per part, divided by the total
// sum of exp) and quantize the result per token into oq (T, H * AHD) FP8, fragment order. One block of 256 threads per
// token, thread = 8 consecutive values = one dim tile of one head (H * AHD == 2048: warp = head).
extern "C" __global__ void combine_quant(const uint64_t *po, const float *st, uint32_t *oq, float *so, int H, int MT, int P) {
    __shared__ float red[8];
    int tok = blockIdx.x, t = threadIdx.x, head = t / (AHD / 8), j = t - head * (AHD / 8), mt = tok >> 4, row = tok & 15;
    float wgt[16], mx = -3.0e38f, den = 0.0f; float2 ms[16];
    #pragma unroll
    for (int p = 0; p < 16; p++) if (p < P) { ms[p] = *(const float2 *)(st + (((size_t)(p * H + head) * MT) * 16 + tok) * 2); mx = fmaxf(mx, ms[p].x); }
    #pragma unroll
    for (int p = 0; p < 16; p++) if (p < P) { wgt[p] = __expf(ms[p].x - mx); den += wgt[p] * ms[p].y; }
    float inv = 1.0f / den, v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t r[16][2];
    #pragma unroll
    for (int p = 0; p < 16; p++) if (p < P) {     // accumulator order: [tile][dim tile][row / 8][(row & 7) * 4 + t] holds dims 2 t, 2 t + 1
        const uint64_t *src = po + (((((size_t)(p * H + head) * MT + mt) * (AHD / 8) + j) * 2 + (row >> 3)) * 32 + (row & 7) * 4) / 2;
        r[p][0] = src[0]; r[p][1] = src[1];
    }
    #pragma unroll
    for (int p = 0; p < 16; p++) if (p < P) {
        const __half *hp = (const __half *)r[p]; float w = wgt[p] * inv;
        #pragma unroll
        for (int i = 0; i < 8; i++) v[i] += w * __half2float(hp[i]);
    }
    float am = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) am = fmaxf(am, fabsf(v[i]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if ((t & 31) == 0) red[t >> 5] = am;
    __syncthreads();
    am = fmaxf(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])), fmaxf(fmaxf(red[4], red[5]), fmaxf(red[6], red[7])));
    int ex = scale_exp(am); float q = pow2f(-ex);
    oq[xfrag(tok, t * 8) >> 2] = pack4(v, q); oq[xfrag(tok, t * 8 + 4) >> 2] = pack4(v + 4, q);       // xfrag: 4 consecutive k = one word
    if (t == 0) so[tok] = pow2f(ex);
}
