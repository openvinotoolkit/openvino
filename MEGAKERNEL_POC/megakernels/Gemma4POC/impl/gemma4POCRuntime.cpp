#include "gemma4POCRuntime.h"

#include <CL/cl_ext.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>



// Shape -D flags are added by the caller (H_, I_, IE_, E_, TOPK_, EPS_).
static const char* kFfnBuildOpts = "-cl-std=CL3.0 -cl-mad-enable";
static constexpr size_t kFfnLws = 256;

// Work-groups of ffn_oproj, ffn_in, ffn_mid, ffn_moe_down, ffn_out; must match
// the *_WGS macros of the source below.
constexpr int kFfnKernels = 5;
inline void ffn_global_sizes(int H, int I, int IE, int E, int topk, size_t gws[kFfnKernels]) {
    gws[0] = kFfnLws * (H / 32);                     // OPROJ_WGS
    gws[1] = kFfnLws * (I / 64 + E / 16);            // DENSE_WGS + ROUTER_WGS
    gws[2] = kFfnLws * (topk * (IE / 64) + H / 32);  // MOE_WGS + DOWN_WGS
    gws[3] = kFfnLws * (H / 32);                     // EDOWN_WGS
    gws[4] = kFfnLws;
}

// OpenCL source of the Gemma-4 FFN-block MegaKernel (see gemma4POCRuntime.h).
// Built with -DH_ -DI_ -DIE_ -DE_ -DTOPK_ -DEPS_ plus kFfnBuildOpts.
//
// int4 GEMV design, each step measured on B60 with research/gemma4_ffn_bench:
//  * v1, 16 lanes per row with f32 dequant: ALU bound at 100-150 GB/s (u8->f32
//    is 3 movs per element on Xe2, scale shuffles used indirect addressing).
//  * v2, one row per lane: the activation becomes uniform across the
//    sub-group and is consumed as a broadcast operand. It is pre-quantised into
//    two int8 planes per 64-element group, x ~= a * (xh + xl / 254), which keeps
//    ~16 bits (more than the f16 the stock graph carries) and turns the inner
//    loop into packed dp4a, 7 integer ops per 8 weights. Zero points use
//    sum((q - z) x) = sum(q x) - z sum(x) with sum(x) over the quantised x, so
//    the identity is exact. Still only ~200 GB/s: each lane gathered its row's
//    32 bytes, scale and zero point from 16 different rows.
//  * v3 (this): weights are repacked at compile time (repack_row_blocks) so that
//    a 16-row block's group is one contiguous 576-byte record - weights plus
//    packed scale|zero point - read with two sub-group block reads.
//  * Row counts too small to fill the GPU are split along K across the
//    sub-groups of a work-group and reduced through SLM.
static const char* kFfnSource = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define SG 16
#define NSG 16
#define LWS (SG * NSG)
#define QG 64
#define XL_RES 254.0f
#define REC_DW 144                            // repacked record: 128 weight + 16 meta dwords

#define REQD __attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(LWS, 1, 1)))

#ifndef GEMV_UNROLL
#define GEMV_UNROLL 4
#endif

// Work-group split; ffn_global_sizes() on the host must match.
#define DENSE_KS 4                            // ffn_in dense: 4 row blocks x 4 K slices
#define DENSE_WGS (I_ / (SG * NSG / DENSE_KS))
#define ROUTER_KS 16                          // ffn_in router: 1 row block x 16 K slices
#define ROUTER_WGS (E_ / SG)
#define MOE_KS 4                              // ffn_mid experts: 4 row blocks x 4 K slices
#define MOE_BLK (IE_ / (SG * NSG / MOE_KS))
#define MOE_WGS (TOPK_ * MOE_BLK)
#define DOWN_KS 8                             // ffn_mid dense down: 2 row blocks x 8 K slices
#define DOWN_WGS (H_ / (SG * NSG / DOWN_KS))
#define EDOWN_RB 2                            // ffn_moe_down: 2 row blocks x TOPK_ slots
#define EDOWN_WGS (H_ / (SG * EDOWN_RB))

// Quantised activation layout (K elements): xh[K/4] | xl[K/4] | xa[K/64] | xs[K/64].
// xh/xl dword 2b = even elements of 8-block b, 2b+1 = odd elements; xa[g] is the
// dequant scale and xs[g] the sum of the dequantised group g.
#define XQ_WORDS(K) ((K) / 2 + 2 * ((K) / QG))

inline float wg_sum(float v, __local float* tmp) {
    v = sub_group_reduce_add(v);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0)
        tmp[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    return sub_group_reduce_add(tmp[get_sub_group_local_id()]);
}

inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

inline uint pack4(int a, int b, int c, int d) {
    return (uint)(a & 255) | ((uint)(b & 255) << 8) | ((uint)(c & 255) << 16) | ((uint)(d & 255) << 24);
}

typedef struct {
    uint he, ho, le, lo;
    float a, gsum;
} QB;

// Quantises the 8 consecutive elements of one 8-block. The 16 lanes of the
// sub-group must hold 16 consecutive blocks starting at a multiple of 16, so
// each 8-lane cluster is one 64-element group; invalid lanes still take part
// in the cluster reductions.
inline QB quant_block(float8 v, bool valid) {
    const float8 av = fabs(v);
    float m = fmax(fmax(fmax(av.s0, av.s1), fmax(av.s2, av.s3)), fmax(fmax(av.s4, av.s5), fmax(av.s6, av.s7)));
    m = sub_group_clustered_reduce_max(valid ? m : 0.0f, 8);
    QB q;
    q.a = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
    const float8 t = v * (1.0f / q.a);
    const float8 hf = rint(t);
    const int8 hi = convert_int8(hf);
    const int8 lo = convert_int8(rint((t - hf) * XL_RES));
    const int sh = sub_group_clustered_reduce_add(valid ? hi.s0 + hi.s1 + hi.s2 + hi.s3 + hi.s4 + hi.s5 + hi.s6 + hi.s7 : 0, 8);
    const int sl = sub_group_clustered_reduce_add(valid ? lo.s0 + lo.s1 + lo.s2 + lo.s3 + lo.s4 + lo.s5 + lo.s6 + lo.s7 : 0, 8);
    q.gsum = q.a * ((float)sh + (float)sl * (1.0f / XL_RES));
    q.he = pack4(hi.s0, hi.s2, hi.s4, hi.s6);
    q.ho = pack4(hi.s1, hi.s3, hi.s5, hi.s7);
    q.le = pack4(lo.s0, lo.s2, lo.s4, lo.s6);
    q.lo = pack4(lo.s1, lo.s3, lo.s5, lo.s7);
    return q;
}

#define DEFINE_STORE_QB(NAME, AS)                                                   \
inline void NAME(QB q, int b, int K, AS uint* x) {                                  \
    x[2 * b] = q.he;                                                                \
    x[2 * b + 1] = q.ho;                                                            \
    x[K / 4 + 2 * b] = q.le;                                                        \
    x[K / 4 + 2 * b + 1] = q.lo;                                                    \
    if ((get_sub_group_local_id() & 7) == 0) {                                      \
        x[K / 2 + (b >> 3)] = as_uint(q.a);                                         \
        x[K / 2 + K / QG + (b >> 3)] = as_uint(q.gsum);                             \
    }                                                                               \
}
DEFINE_STORE_QB(store_qb_l, __local)
DEFINE_STORE_QB(store_qb_g, __global)

// Quantises x[0..K) (global, loaded with LOAD8(b) as float8) into SLM;
// executed by the whole work-group.
#define QUANT_GLOBAL_WITH(LOAD8, K, xq)                                             \
    for (int c0_ = get_sub_group_id() * SG; c0_ < (K) / 8; c0_ += NSG * SG) {       \
        const int b_ = c0_ + get_sub_group_local_id();                             \
        const bool ok_ = b_ < (K) / 8;                                             \
        const QB q_ = quant_block(ok_ ? LOAD8(b_) : (float8)(0.0f), ok_);          \
        if (ok_)                                                                   \
            store_qb_l(q_, b_, (K), (xq));                                         \
    }

// One group of 64 weights (8 dwords of this lane's row) against the group's
// broadcast activation dwords.
#ifdef DOT_TRIVIAL  // harness-only: keeps every load, drops the math
#define DOT8(wv, xhv, xlv, ih, il) \
    ih += (int)((wv).s0 ^ (wv).s1 ^ (wv).s2 ^ (wv).s3 ^ (wv).s4 ^ (wv).s5 ^ (wv).s6 ^ (wv).s7 ^ (xhv) ^ (xlv));
#else
#define DOT8(wv, xhv, xlv, ih, il)                                                          \
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {                       \
        const uint lo = (wv)[j] & 0x0F0F0F0Fu;                                              \
        const uint hi = ((wv)[j] >> 4) & 0x0F0F0F0Fu;                                       \
        ih += dot_4x8packed_us_int(lo, sub_group_broadcast(xhv, 2 * j)) +                   \
              dot_4x8packed_us_int(hi, sub_group_broadcast(xhv, 2 * j + 1));                \
        il += dot_4x8packed_us_int(lo, sub_group_broadcast(xlv, 2 * j)) +                   \
              dot_4x8packed_us_int(hi, sub_group_broadcast(xlv, 2 * j + 1));                \
    }
#endif

// NR row blocks per sub-group (lane l = row l of each block, sharing the
// activation) over ng groups. rec[r] is the block's record of its first group
// (group g0 of the activation xq of K elements, in address space AS).
#define DEFINE_ROWDOT(NAME, NR, AS)                                                         \
inline void NAME(const __global uint* rec[NR], const int g0, const int ng,                  \
                 const AS uint* xq, const int K, float acc[NR]) {                           \
    const AS uint* xh = xq;                                                                 \
    const AS uint* xl = xq + K / 4;                                                         \
    const AS float* xa = (const AS float*)(xq + K / 2);                                     \
    const AS float* xs = xa + K / QG;                                                       \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < NR; ++r) acc[r] = 0.0f;        \
    __attribute__((opencl_unroll_hint(GEMV_UNROLL))) for (int gi = 0; gi < ng; ++gi) {      \
        const int g = g0 + gi;                                                              \
        const uint xhv = intel_sub_group_block_read(xh + g * 16);                           \
        const uint xlv = intel_sub_group_block_read(xl + g * 16);                           \
        const float a = xa[g];                                                              \
        const float sx = xs[g];                                                             \
        __attribute__((opencl_unroll_hint)) for (int r = 0; r < NR; ++r) {                  \
            const __global uint* p = rec[r] + gi * REC_DW;                                  \
            const uint8 wv = intel_sub_group_block_read8(p);                                \
            const uint meta = intel_sub_group_block_read(p + 128);                          \
            int ih = 0, il = 0;                                                             \
            DOT8(wv, xhv, xlv, ih, il)                                                      \
            const float sc = (float)as_half2(meta).s0;                                      \
            const float zz = (float)(meta >> 16);                                           \
            acc[r] = fma(sc, fma(a, fma((float)il, 1.0f / XL_RES, (float)ih), -zz * sx), acc[r]); \
        }                                                                                   \
    }                                                                                       \
}

DEFINE_ROWDOT(rowdot1, 1, __local)
DEFINE_ROWDOT(rowdot2, 2, __local)
DEFINE_ROWDOT(rowdot1g, 1, __global)
DEFINE_ROWDOT(rowdot2g, 2, __global)

// Weight records loaded ahead of a kernel's activation prologue, so their DRAM
// latency overlaps the RMS/quantisation work instead of following it.
#ifndef PRE_G
#define PRE_G 4
#endif
typedef struct {
    uint8 w[PRE_G];
    uint m[PRE_G];
} PreRec;

inline PreRec preload(const __global uint* rec, int ng) {
    PreRec p;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < PRE_G; ++i) {
        if (i < ng) {
            p.w[i] = intel_sub_group_block_read8(rec + i * REC_DW);
            p.m[i] = intel_sub_group_block_read(rec + i * REC_DW + 128);
        }
    }
    return p;
}

// acc += the first min(PRE_G, ng) groups from `p`; the caller continues with
// rowdot over the remaining groups.
inline float predot(const PreRec* p, const int g0, const int ng, const __local uint* xq, const int K) {
    const __local uint* xh = xq;
    const __local uint* xl = xq + K / 4;
    const __local float* xa = (const __local float*)(xq + K / 2);
    const __local float* xs = xa + K / QG;
    float acc = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int gi = 0; gi < PRE_G; ++gi) {
        if (gi < ng) {
            const int g = g0 + gi;
            const uint xhv = intel_sub_group_block_read(xh + g * 16);
            const uint xlv = intel_sub_group_block_read(xl + g * 16);
            const uint8 wv = p->w[gi];
            int ih = 0, il = 0;
            DOT8(wv, xhv, xlv, ih, il)
            const float sc = (float)as_half2(p->m[gi]).s0;
            const float zz = (float)(p->m[gi] >> 16);
            acc = fma(sc, fma(xa[g], fma((float)il, 1.0f / XL_RES, (float)ih), -zz * xs[g]), acc);
        }
    }
    return acc;
}

// Record of row block rb, group g in a repacked [rows, K] matrix.
inline const __global uint* rec_at(const __global uint* m, int K, int rb, int g) {
    return m + ((size_t)rb * (K / QG) + g) * REC_DW;
}

// A work-group that produced the 64 outputs of quantisation group g (in SLM)
// quantises them straight into the next kernel's activation buffer, so that
// kernel needs no SLM prologue or barrier. Called by one full sub-group.
inline void quant_group_store(const __local float* vals, int g, int K, __global uint* xq) {
    const int ln = get_sub_group_local_id();
    const bool ok = ln < 8;
    const QB q = quant_block(ok ? vload8(ln, vals) : (float8)(0.0f), ok);
    if (ok)
        store_qb_g(q, g * 8 + ln, K, xq);
}

// Softmax over the top-k router logits, computed by one sub-group; ties go to
// the lower expert index, as in the stock MoERouterFused kernel. Returns the
// expert of slot `want`; with store set, writes all slots out.
inline int topk_softmax(const __global float* logits, int want, bool store, __global int* tk_idx,
                        __global float* tk_w) {
    const int ln = get_sub_group_local_id();
    float v[E_ / SG];
    // NaN logits must not produce an out-of-range expert index.
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i) {
        const float l = logits[ln + SG * i];
        v[i] = isnan(l) ? -INFINITY : l;
    }
    float top = 0.0f, wsum = 0.0f, mine = 0.0f;
    int mine_idx = 0, wanted = 0;
    __attribute__((opencl_unroll_hint)) for (int t = 0; t < TOPK_; ++t) {
        float lm = v[0];
        int li = ln;
        __attribute__((opencl_unroll_hint)) for (int i = 1; i < E_ / SG; ++i) {
            if (v[i] > lm) {
                lm = v[i];
                li = ln + SG * i;
            }
        }
        const float mx = sub_group_reduce_max(lm);
        const int idx = sub_group_reduce_min(lm == mx ? li : 0x7fffffff);
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i)
            if (ln + SG * i == idx)
                v[i] = -INFINITY;
        if (t == 0)
            top = mx;
        const float e = exp(mx - top);
        wsum += e;
        if (ln == t) {
            mine = e;
            mine_idx = idx;
        }
        if (t == want)
            wanted = idx;
    }
    if (store && ln < TOPK_) {
        tk_idx[ln] = mine_idx;
        tk_w[ln] = mine / wsum;
    }
    return wanted;
}

// ---------------------------------------------------------------------------
// ffn_oproj: o = o_proj(attn), the attention output projection ([H, KA], KA =
// heads * head_dim differs between sliding and full-attention layers).
// ---------------------------------------------------------------------------
#define OPROJ_KMAX 8192
#define OPROJ_KS 8                            // 2 row blocks x 8 K slices
#define OPROJ_WGS (H_ / (SG * NSG / OPROJ_KS))

REQD __kernel void ffn_oproj(const __global half* attn, const __global uint* opk, const int KA,
                             __global float* o_out) {
    __local uint xq[XQ_WORDS(OPROJ_KMAX)];
    __local float red[NSG][SG];
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int ks = sg % OPROJ_KS;
    const int rb = get_group_id(0) * (NSG / OPROJ_KS) + sg / OPROJ_KS;
    const int G = KA / QG;
    const int g0 = ks * G / OPROJ_KS;
    const int g1 = (ks + 1) * G / OPROJ_KS;
    const __global uint* rec0 = rec_at(opk, KA, rb, g0);
#if PRE_G
    const PreRec pre = preload(rec0, g1 - g0);
#endif
#define LOAD_ATTN(b) vload_half8((b), attn)
    QUANT_GLOBAL_WITH(LOAD_ATTN, KA, xq)
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[1];
#if PRE_G
    const __global uint* rec[1] = {rec0 + PRE_G * REC_DW};
    rowdot1(rec, g0 + PRE_G, g1 - g0 - PRE_G, xq, KA, acc);
    acc[0] += predot(&pre, g0, g1 - g0, xq, KA);
#else
    const __global uint* rec[1] = {rec0};
    rowdot1(rec, g0, g1 - g0, xq, KA, acc);
#endif
    red[sg][ln] = acc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float v = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < OPROJ_KS; ++k)
            v += red[sg + k][ln];
        o_out[rb * SG + ln] = v;
    }
}

// ---------------------------------------------------------------------------
// ffn_in: h1 = h + RMS(o)*w_pa, then
//   dense WGs:  a = gelu(gate(x)) * up(x),  x = RMS(h1)*w_pf, written quantised
//   router WGs: logits = Wr . (RMS(h1)*w_r)
// WG 0 also stores h1 and the quantised MoE input RMS(h1)*w_pf2.
// ---------------------------------------------------------------------------
#define HB (H_ / 8)
#define HB_PER_LANE ((HB + LWS - 1) / LWS)

REQD __kernel void ffn_in(const __global float* o, const __global half* h,
                          const __global half* w_pa, const __global half* w_pf,
                          const __global half* w_r, const __global half* w_pf2,
                          const __global uint* gpk, const __global uint* upk,
                          const __global half* rw,
                          __global float* h1_out, __global uint* x2q,
                          __global uint* actq, __global float* logits_out) {
    __local uint xq[XQ_WORDS(H_)];
    __local float xf[H_];
    __local float red[NSG][SG * 2];
    __local float outv[NSG / DENSE_KS * SG];
    __local float tmp[NSG];
    const int wg = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const bool router = wg >= DENSE_WGS;

    // Work-item handles 8-element blocks sg*16 + ln + 256*i.
    float8 hv[HB_PER_LANE];
    float acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        hv[i] = b < HB ? vload8(b, o) : (float8)(0.0f);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv_o = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        if (b < HB)
            hv[i] = vload_half8(b, h) + hv[i] * inv_o * vload_half8(b, w_pa);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv1 = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        const bool ok = b < HB;
        if (router) {
            if (ok)
                vstore8(hv[i] * inv1 * vload_half8(b, w_r), b, xf);
        } else {
            const QB q = quant_block(ok ? hv[i] * inv1 * vload_half8(b, w_pf) : (float8)(0.0f), ok);
            if (ok)
                store_qb_l(q, b, H_, xq);
        }
        if (wg == 0) {
            const QB q = quant_block(ok ? hv[i] * inv1 * vload_half8(b, w_pf2) : (float8)(0.0f), ok);
            if (ok) {
                store_qb_g(q, b, H_, x2q);
                vstore8(hv[i], b, h1_out);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (!router) {
        const int ks = sg % DENSE_KS;
        const int rbl = sg / DENSE_KS;
        const int rb = wg * (NSG / DENSE_KS) + rbl;
        const int gps = (H_ / QG) / DENSE_KS;
        const __global uint* rec[2] = {rec_at(gpk, H_, rb, ks * gps), rec_at(upk, H_, rb, ks * gps)};
        float acc[2];
        rowdot2(rec, ks * gps, gps, xq, H_, acc);
        red[sg][ln] = acc[0];
        red[sg][SG + ln] = acc[1];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float g = 0.0f, u = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < DENSE_KS; ++k) {
                g += red[sg + k][ln];
                u += red[sg + k][SG + ln];
            }
            outv[rbl * SG + ln] = gelu_tanh(g) * u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0)
            quant_group_store(outv, wg, I_, actq);
    } else {
        // f16 router weights, one row per lane, K split 16 ways.
        const int n = (wg - DENSE_WGS) * SG + ln;
        const int k0 = sg * (H_ / ROUTER_KS);
        const __global half* row = rw + n * H_ + k0;
        float acc = 0.0f;
        __attribute__((opencl_unroll_hint(4))) for (int c = 0; c < (H_ / ROUTER_KS) / SG; ++c) {
            const float16 w = vload_half16(c, row);
            const float x = as_float(intel_sub_group_block_read((const __local uint*)(xf + k0 + c * SG)));
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < SG; ++j)
                acc = fma(w[j], sub_group_broadcast(x, j), acc);
        }
        red[sg][ln] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0) {
            float v = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < ROUTER_KS; ++k)
                v += red[k][ln];
            logits_out[n] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// ffn_mid: MoE WGs:  slot t of the top-k: moe_act[t] = gelu(gate_e(x2)) * up_e(x2),
//                    written quantised for ffn_moe_down
//          down WGs: m = down(a)
// The activations come pre-quantised from global memory, so the GEMV starts
// without any SLM prologue or barrier.
// ---------------------------------------------------------------------------
REQD __kernel void ffn_mid(const __global uint* x2q, const __global float* logits, const __global uint* actq,
                           const __global uint* egpk, const __global uint* eupk, const __global uint* dpk,
                           __global uint* moe_actq, __global float* m_out,
                           __global int* tk_idx_out, __global float* tk_w_out) {
    __local float red[NSG][SG * 2];
    __local float outv[NSG / MOE_KS * SG];
    const int wg = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();

    if (wg < MOE_WGS) {
        const int ks = sg % MOE_KS;
        const int rbl = sg / MOE_KS;
        const int slot = wg / MOE_BLK;
        const int blk = wg % MOE_BLK;
        // Every sub-group derives its expert itself: cheaper than a barrier.
        const int expert = topk_softmax(logits, slot, wg == 0 && sg == 0, tk_idx_out, tk_w_out);
        const int nb = blk * (NSG / MOE_KS) + rbl;   // row block within the expert
        const int rb = expert * (IE_ / SG) + nb;
        const int gps = (H_ / QG) / MOE_KS;
        const __global uint* rec[2] = {rec_at(egpk, H_, rb, ks * gps), rec_at(eupk, H_, rb, ks * gps)};
        float acc[2];
        rowdot2g(rec, ks * gps, gps, x2q, H_, acc);
        red[sg][ln] = acc[0];
        red[sg][SG + ln] = acc[1];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float g = 0.0f, u = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < MOE_KS; ++k) {
                g += red[sg + k][ln];
                u += red[sg + k][SG + ln];
            }
            outv[rbl * SG + ln] = gelu_tanh(g) * u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg == 0)
            quant_group_store(outv, slot * MOE_BLK + blk, TOPK_ * IE_, moe_actq);
    } else {
        const int ks = sg % DOWN_KS;
        const int rb = (wg - MOE_WGS) * (NSG / DOWN_KS) + sg / DOWN_KS;
        const int g0 = ks * (I_ / QG) / DOWN_KS;
        const int g1 = (ks + 1) * (I_ / QG) / DOWN_KS;
        const __global uint* rec[1] = {rec_at(dpk, I_, rb, g0)};
        float acc[1];
        rowdot1g(rec, g0, g1 - g0, actq, I_, acc);
        red[sg][ln] = acc[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) {
            float v = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int k = 0; k < DOWN_KS; ++k)
                v += red[sg + k][ln];
            m_out[rb * SG + ln] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// ffn_moe_down: e[n] = sum_t w_t * down_{e_t}[n] . moe_act[t]
// Sub-group sg handles slot sg % TOPK_ of row block sg / TOPK_.
// ---------------------------------------------------------------------------
REQD __kernel void ffn_moe_down(const __global uint* moe_actq, const __global int* tk_idx,
                                const __global float* tk_w, const __global uint* edpk,
                                __global float* e_out) {
    __local float red[NSG][SG];
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int t = sg % TOPK_;
    const int nb = get_group_id(0) * EDOWN_RB + sg / TOPK_;
    const int rb = tk_idx[t] * (H_ / SG) + nb;
    const __global uint* rec[1] = {rec_at(edpk, IE_, rb, 0)};
    float acc[1];
    rowdot1g(rec, t * (IE_ / QG), IE_ / QG, moe_actq, TOPK_ * IE_, acc);
    red[sg][ln] = tk_w[t] * acc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) {
        float v = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < TOPK_; ++k)
            v += red[sg + k][ln];
        e_out[nb * SG + ln] = v;
    }
}

// ---------------------------------------------------------------------------
// ffn_out (one WG): out = (h1 + RMS(RMS(m)*w1 + RMS(e)*w2) * w_post) * layer_scalar
// ---------------------------------------------------------------------------
#define HPT (H_ / LWS)
REQD __kernel void ffn_out(const __global float* h1, const __global float* m, const __global float* e,
                           const __global half* w1, const __global half* w2, const __global half* w_post,
                           const __global half* layer_scalar, __global half* out) {
    __local float tmp[NSG];
    const int lid = get_local_id(0);
    float mv[HPT], ev[HPT];
    float sm = 0.0f, se = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        mv[i] = m[lid + i * LWS];
        ev[i] = e[lid + i * LWS];
        sm += mv[i] * mv[i];
        se += ev[i] * ev[i];
    }
    const float inv_m = rsqrt(wg_sum(sm, tmp) / H_ + EPS_);
    const float inv_e = rsqrt(wg_sum(se, tmp) / H_ + EPS_);
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        mv[i] = mv[i] * inv_m * vload_half(k, w1) + ev[i] * inv_e * vload_half(k, w2);
        ss += mv[i] * mv[i];
    }
    const float inv_s = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
    const float ls = vload_half(0, layer_scalar);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        vstore_half((h1[k] + mv[i] * inv_s * vload_half(k, w_post)) * ls, k, out);
    }
}

// ---------------------------------------------------------------------------
// attn_qkv: raw = Wqkv . (RMS(x) * w_in), the stacked q|k|v projection rows.
// ---------------------------------------------------------------------------
#define QKV_KS 4                              // 4 row blocks x 4 K slices
REQD __kernel void attn_qkv(const __global half* x, const __global half* w_in, const __global uint* wpk,
                            __global float* raw) {
    __local uint xq[XQ_WORDS(H_)];
    __local float red[NSG][SG];
    __local float tmp[NSG];
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int ks = sg % QKV_KS;
    const int rb = get_group_id(0) * (NSG / QKV_KS) + sg / QKV_KS;
    const int g0 = ks * (H_ / QG) / QKV_KS;
    const int g1 = (ks + 1) * (H_ / QG) / QKV_KS;
    const __global uint* rec0 = rec_at(wpk, H_, rb, g0);
#if PRE_G
    const PreRec pre = preload(rec0, g1 - g0);
#endif

    float8 hv[HB_PER_LANE];
    float acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        hv[i] = b < HB ? vload_half8(b, x) : (float8)(0.0f);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        const bool ok = b < HB;
        const QB q = quant_block(ok ? hv[i] * inv * vload_half8(b, w_in) : (float8)(0.0f), ok);
        if (ok)
            store_qb_l(q, b, H_, xq);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float acc[1];
#if PRE_G
    const __global uint* rec[1] = {rec0 + PRE_G * REC_DW};
    rowdot1(rec, g0 + PRE_G, g1 - g0 - PRE_G, xq, H_, acc);
    acc[0] += predot(&pre, g0, g1 - g0, xq, H_);
#else
    const __global uint* rec[1] = {rec0};
    rowdot1(rec, g0, g1 - g0, xq, H_, acc);
#endif
    red[sg][ln] = acc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ks == 0) {
        float v = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < QKV_KS; ++k)
            v += red[sg + k][ln];
        raw[rb * SG + ln] = v;
    }
}

// ---------------------------------------------------------------------------
// attn_post: one sub-group per head. q and k heads: RMS * gamma, then
// rotate-half RoPE over the leading `rn` dims; v heads: gamma-less RMS.
// meta = {HQ, HK, D, rn, raw k row, raw v row, raw q row, position stride
// (ints), position rounded to half, table width C}. The RoPE tables are those of
// the exported rotary_emb: cos/sin(half(pos) * inv_freq[i % (C/2)]) in f32,
// stored as half. out is head-major [B, HQ+2*HK, S, D]; t = b * S + s.
// Token t = t0 + get_group_id(1) reads raw + get_group_id(1) * rstride.
// ---------------------------------------------------------------------------
#define ATTN_DMAX 512
REQD __kernel void attn_post(const __global float* raw, const __global half* gq, const __global half* gk,
                             const __global int* meta, const __global int* pos, const __global float* invf,
                             __global half* out_base, const int t0, const int S, const int rstride) {
    __local float sv[NSG][ATTN_DMAX];
    __local float ctab[ATTN_DMAX], stab[ATTN_DMAX];
    const int t = t0 + get_group_id(1);
    raw += (size_t)get_group_id(1) * rstride;
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int head = get_group_id(0) * NSG + sg;
    const int HQ = meta[0], HK = meta[1], D = meta[2], RN = meta[3];
    if (get_group_id(0) * NSG >= HQ + 2 * HK)
        return;
    {
        const int half_c = meta[9] / 2;
        float p = (float)pos[(size_t)t * meta[7]];
        if (meta[8])
            p = (float)convert_half_rte(p);
        for (int i = get_local_id(0); i < RN; i += LWS) {
            const float f = p * invf[i % half_c];
            ctab[i] = (float)convert_half_rte(cos(f));
            stab[i] = (float)convert_half_rte(sin(f));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (head >= HQ + 2 * HK)
        return;
    const int HT = HQ + 2 * HK;
    __global half* out = out_base + ((size_t)((t / S) * HT + head) * S + t % S) * D;
    const int dst = 0;
    const __global half* g = 0;
    if (head < HQ) {
        raw += meta[6] + head * D;
        g = gq;
    } else if (head < HQ + HK) {
        raw += meta[4] + (head - HQ) * D;
        g = gk;
    } else {
        raw += meta[5] + (head - HQ - HK) * D;
    }
    const int nj = D / SG;
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < ATTN_DMAX / SG; ++j) {
        if (j < nj) {
            const float v = raw[j * SG + ln];
            sv[sg][j * SG + ln] = v;
            ss += v * v;
        }
    }
    const float r = rsqrt(sub_group_reduce_add(ss) / D + EPS_);
    if (g) {
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < ATTN_DMAX / SG; ++j)
            if (j < nj)
                sv[sg][j * SG + ln] *= r * vload_half(j * SG + ln, g);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        const int half_rn = RN / 2;
        for (int i = ln; i < D; i += SG) {
            float y = sv[sg][i];
            if (i < half_rn)
                y = y * ctab[i] - sv[sg][i + half_rn] * stab[i];
            else if (i < RN)
                y = y * ctab[i] + sv[sg][i - half_rn] * stab[i];
            vstore_half(y, dst + i, out);
        }
    } else {
        for (int i = ln; i < D; i += SG)
            vstore_half(sv[sg][i] * r, dst + i, out);
    }
}

// ---------------------------------------------------------------------------
// Attention masks [B,1,S,L] from attention_mask [B,L] and token_type_ids [B,S]
// (i32 or i64: mm[8] / mm[9] ints per element). Query i sits at L - S + i.
// mm = {fill, -, pad_mul, pad_add, window, causal_off, vision_id0, vision_id1}:
//   full    = (j >= q + causal_off ? fill : 0) + am[j] * pad_mul + pad_add
//   sliding = same vision group ? 0 : (q - j >= window ? fill : full)
// Vision groups number the runs of vision tokens. As exported, token types are
// padded on the right to L, so positions >= S count as text.
// ---------------------------------------------------------------------------
inline int vis_at(const __global int* tti, int istride, int S, int p, float v0, float v1) {
    if (p >= S)
        return 0;
    const float t = (float)tti[(size_t)p * istride];
    return t == v0 || t == v1;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void mask_groups(const __global int* tti, const __global float* mm, const int S, const int L,
                          __global int* grp) {
    __local int carry;
    const int istride = (int)mm[9];
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    tti += (size_t)b * S * istride;
    grp += (size_t)b * L;
    if (lid == 0)
        carry = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p0 = 0; p0 < L; p0 += 256) {
        const int p = p0 + lid;
        const int vis = p < L ? vis_at(tti, istride, S, p, mm[6], mm[7]) : 0;
        const int prev = p > 0 && p - 1 < L ? vis_at(tti, istride, S, p - 1, mm[6], mm[7]) : 0;
        const int base = carry;
        const int run = work_group_scan_inclusive_add(vis & !prev);
        if (p < L)
            grp[p] = vis ? base + run - 1 : -1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 255)
            carry = base + run;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void mask_fill(const __global int* am, const __global float* mm, const __global int* grp, const int S,
                        const int L, const int sliding, __global half* out) {
    const int istride = (int)mm[8];
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    const int b = get_global_id(2);
    if (j >= L)
        return;
    const int q = L - S + i;
    const float a = (float)am[((size_t)b * L + j) * istride];
    float v = (j >= q + (int)mm[5] ? mm[0] : 0.0f) + fma(a, mm[2], mm[3]);
    if (sliding) {
        const int gj = grp[(size_t)b * L + j];
        if (gj >= 0 && gj == grp[(size_t)b * L + q])
            v = 0.0f;
        else if (q - j >= (int)mm[4])
            v = mm[0];
    }
    vstore_half_rte(v, ((size_t)b * S + i) * L + j, out);
}
)CLC";

constexpr int kAttnQkvRowsPerWg = 64;  // SG * NSG / QKV_KS



// Multi-token (prefill) path of the Gemma-4 MegaKernels. Appended to kFfnSource
// (it reuses REC_DW, rec_at, wg_sum, gelu_tanh, topk_softmax) and built as a
// separate program with the large register file.
//
// The decode kernels are GEMVs that re-read every weight once per token. Here
// all tokens of a call go through int4 XMX GEMMs on the same repacked records:
//  * One sub-group owns PF_MT tokens x 2 row blocks (32 rows). A record is
//    block-read once per 64-group and dequantised to f16 in registers, then fed
//    as the DPAS B operand to PF_MB 8-token A tiles.
//  * Dequantisation is a mask + or per dword: (w >> 4j) & 0x000F000F lands the
//    nibbles of k and k+4 in the two halves, and 0x6400 turns them into 1024+q.
//    Activations are therefore stored row-major with each 16-chunk permuted to
//    match ("P order": position l holds k = 8*(l/8) + (l%8)/2 + 4*(l%2)), and
//    the A tiles come straight from 2D block reads, which also zero the rows
//    past the end.
//  * The MoE runs as a grouped GEMM: (token, slot) pairs are bucketed per
//    expert into PF_MT-row tiles, gathered (plain row copies), and the down
//    projection scatters its weighted rows back per pair.
static const char* kPrefillBuildOpts = "-cl-intel-256-GRF-per-thread";

#ifndef PF_MB_DEFAULT
#define PF_MB_DEFAULT 8
#endif
// The program is built twice: kPfMb for the dense GEMMs and well-filled expert
// tiles, kPfMbSmall for short prompts and experts that get only a few rows.
constexpr int kPfMb = PF_MB_DEFAULT;    // 8-token blocks per sub-group
constexpr int kPfMt = 8 * kPfMb;        // tokens per GEMM tile
constexpr int kPfMbSmall = 1;
constexpr int kPfNsg = 8;               // sub-groups per GEMM work-group
constexpr size_t kPfLws = 16 * kPfNsg;
constexpr int kPfRouterKs = 11;         // router K slices (H / 256)

static const char* kPrefillSource = R"CLC(
#define PF_MT (8 * PF_MB)
#define PF_LWS (SG * PF_NSG)
#define PF_REQD __attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(PF_LWS, 1, 1)))

// Stores elements k = 8b..8b+7 of a P-order row.
inline void pf_store8(float8 v, int b, __global half* row) {
    vstore8(convert_half8((float8)(v.s0, v.s4, v.s1, v.s5, v.s2, v.s6, v.s3, v.s7)), b, row);
}

// DPAS B operand (16 k x 16 rows, one row per lane) from the two record
// dwords holding k 0..7 and 8..15 of this lane's row.
inline int8 pf_deq(uint lo, uint hi, half2 s, half2 z) {
    const uint M = 0x000F000Fu, B = 0x64006400u;
    int8 r;
    r.s0 = as_int((as_half2((lo & M) | B) - z) * s);
    r.s1 = as_int((as_half2(((lo >> 4) & M) | B) - z) * s);
    r.s2 = as_int((as_half2(((lo >> 8) & M) | B) - z) * s);
    r.s3 = as_int((as_half2(((lo >> 12) & M) | B) - z) * s);
    r.s4 = as_int((as_half2((hi & M) | B) - z) * s);
    r.s5 = as_int((as_half2(((hi >> 4) & M) | B) - z) * s);
    r.s6 = as_int((as_half2(((hi >> 8) & M) | B) - z) * s);
    r.s7 = as_int((as_half2(((hi >> 12) & M) | B) - z) * s);
    return r;
}

// The PF_MB A tiles (8 rows x 16 k each) of rows row0.. at k chunk kc of a
// P-order [rows, K] activation; rows >= `rows` read as zero.
#if PF_MB % 4 == 0
#define PF_LOAD_A(av, at, K, rows, row0, kc)                                                        \
    __attribute__((opencl_unroll_hint)) for (int q_ = 0; q_ < PF_MB / 4; ++q_)                     \
        intel_sub_group_2d_block_read_16b_32r16x1c((__global void*)(at), (K) * 2, (rows), (K) * 2,  \
                                                   (int2)((kc) * 16, (row0) + q_ * 32),             \
                                                   (__private ushort*)&av[q_ * 4]);
#else
#define PF_LOAD_A(av, at, K, rows, row0, kc)                                                        \
    __attribute__((opencl_unroll_hint)) for (int q_ = 0; q_ < PF_MB; ++q_)                         \
        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)(at), (K) * 2, (rows), (K) * 2,   \
                                                  (int2)((kc) * 16, (row0) + q_ * 8),               \
                                                  (__private ushort*)&av[q_]);
#endif

// acc0/acc1[mb] += A(rows row0.., K) x records r0/r1 (two 16-row blocks).
#define PF_CORE(at, K, rows, row0, r0, r1, acc0, acc1)                                             \
    __attribute__((opencl_unroll_hint(1))) for (int g = 0; g < (K) / QG; ++g) {                   \
        const uint8 w0 = intel_sub_group_block_read8(r0 + g * REC_DW);                             \
        const uint m0 = intel_sub_group_block_read(r0 + g * REC_DW + 128);                         \
        const uint8 w1 = intel_sub_group_block_read8(r1 + g * REC_DW);                             \
        const uint m1 = intel_sub_group_block_read(r1 + g * REC_DW + 128);                         \
        const half2 s0 = (half2)(as_half2(m0).s0), z0 = (half2)((half)(1024 + (m0 >> 16)));        \
        const half2 s1 = (half2)(as_half2(m1).s0), z1 = (half2)((half)(1024 + (m1 >> 16)));        \
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < 4; ++c) {                          \
            ushort8 av_[PF_MB];                                                                    \
            PF_LOAD_A(av_, at, K, rows, row0, g * 4 + c)                                           \
            const int8 b0 = pf_deq(w0[2 * c], w0[2 * c + 1], s0, z0);                              \
            __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)                 \
                acc0[mb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(av_[mb]), b0, acc0[mb]); \
            const int8 b1 = pf_deq(w1[2 * c], w1[2 * c + 1], s1, z1);                              \
            __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)                 \
                acc1[mb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(av_[mb]), b1, acc1[mb]); \
        }                                                                                          \
    }

#define PF_ZERO(acc) __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) acc[mb] = (float8)(0.0f);

// Column (lane) whose value sits at position l of a P-order 16-chunk.
inline int pf_pos_lane(int l) { return (l & 8) | ((l & 7) >> 1) | ((l & 1) << 2); }

// Writes acc (rows row0.., the sub-group's 16 columns = chunk c) as P-order f16.
inline void pf_store_rows(const float8 acc[PF_MB], int row0, int c, int K, int rows, __global half* out) {
    const int src = pf_pos_lane(get_sub_group_local_id());
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        ushort8 v;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
            v[m] = as_ushort(convert_half(sub_group_shuffle(acc[mb][m], src)));
        intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)out, K * 2, rows, K * 2,
                                                   (int2)(c * 16, row0 + mb * 8), (__private ushort*)&v);
    }
}

// ---------------------------------------------------------------------------
// out[M, N] (f32, row stride ldo) = A[M, K] . W^T. Sub-group j: row blocks 2j, 2j+1.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm(const __global half* at, const int K, const __global uint* w, const int nrb,
                              const int M, __global float* out, const int ldo) {
    const int j = get_group_id(0) * PF_NSG + get_sub_group_id();
    if (2 * j >= nrb)
        return;
    const int row0 = get_group_id(1) * PF_MT;
    const int ln = get_sub_group_local_id();
    const __global uint* r0 = rec_at(w, K, 2 * j, 0);
    const __global uint* r1 = rec_at(w, K, 2 * j + 1, 0);
    float8 acc0[PF_MB], acc1[PF_MB];
    PF_ZERO(acc0) PF_ZERO(acc1)
    PF_CORE(at, K, M, row0, r0, r1, acc0, acc1)
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int row = row0 + mb * 8 + m;
            if (row < M) {
                out[(size_t)row * ldo + j * 32 + ln] = acc0[mb][m];
                out[(size_t)row * ldo + j * 32 + 16 + ln] = acc1[mb][m];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// act = gelu(gate . x) * (up . x), P-order f16 for the down projection. With
// tile_exp set (MoE), M tile t uses expert tile_exp[t] and only the first
// hdr[0] tiles exist; otherwise there are M rows.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm_gelu(const __global half* at, const int K, const __global uint* wg,
                                   const __global uint* wu, const int nrb, const int M, const __global int* hdr,
                                   const __global int* tile_exp, __global half* act) {
    const int rb = get_group_id(0) * PF_NSG + get_sub_group_id();
    const int mt = get_group_id(1);
    const int rows = hdr ? hdr[0] * PF_MT : M;
    if (rb >= nrb || mt * PF_MT >= rows)
        return;
    const int erb = (tile_exp ? tile_exp[mt] * nrb : 0) + rb;
    const __global uint* r0 = rec_at(wg, K, erb, 0);
    const __global uint* r1 = rec_at(wu, K, erb, 0);
    float8 acc0[PF_MB], acc1[PF_MB];
    PF_ZERO(acc0) PF_ZERO(acc1)
    PF_CORE(at, K, rows, mt * PF_MT, r0, r1, acc0, acc1)
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
            acc0[mb][m] = gelu_tanh(acc0[mb][m]) * acc1[mb][m];
    }
    pf_store_rows(acc0, mt * PF_MT, rb, nrb * 16, rows, act);
}

// ---------------------------------------------------------------------------
// MoE down projection: ys[pair] = w[pair] * (down_e . act[row]) for every
// gathered row; padded rows (pair < 0) are dropped.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm_moe_down(const __global half* at, const int K, const __global uint* w,
                                       const int nrb, const __global int* hdr, const __global int* tile_exp,
                                       const __global int* pair_of_row, const __global float* tk_w,
                                       __global half* ys) {
    const int j = get_group_id(0) * PF_NSG + get_sub_group_id();
    const int mt = get_group_id(1);
    const int rows = hdr[0] * PF_MT;
    if (2 * j >= nrb || mt * PF_MT >= rows)
        return;
    const int ln = get_sub_group_local_id();
    const int erb = tile_exp[mt] * nrb + 2 * j;
    const __global uint* r0 = rec_at(w, K, erb, 0);
    const __global uint* r1 = rec_at(w, K, erb + 1, 0);
    float8 acc0[PF_MB], acc1[PF_MB];
    PF_ZERO(acc0) PF_ZERO(acc1)
    PF_CORE(at, K, rows, mt * PF_MT, r0, r1, acc0, acc1)
    const int N = nrb * 16;
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int p = pair_of_row[mt * PF_MT + mb * 8 + m];
            if (p >= 0) {
                const float tw = tk_w[p];
                ys[(size_t)p * N + j * 32 + ln] = convert_half(tw * acc0[mb][m]);
                ys[(size_t)p * N + j * 32 + 16 + ln] = convert_half(tw * acc1[mb][m]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Router logits, partial over K slice get_group_id(0):
//   out[ks][M][N] = A[:, slice] . W[:, slice]^T, dense f16 W [N, K], N = 16 * PF_NSG.
// ---------------------------------------------------------------------------
PF_REQD __kernel void pf_gemm_f16w(const __global half* at, const int K, const __global half* w, const int N,
                                   const int M, __global float* out) {
    const int rb = get_sub_group_id();
    const int ks = get_group_id(0);
    const int nks = get_num_groups(0);
    const int row0 = get_group_id(1) * PF_MT;
    const int ln = get_sub_group_local_id();
    const __global half* row = w + (size_t)(rb * 16 + ln) * K;
    const int c0 = ks * (K >> 4) / nks, c1 = (ks + 1) * (K >> 4) / nks;
    float8 acc[PF_MB];
    PF_ZERO(acc)
    __attribute__((opencl_unroll_hint(2))) for (int c = c0; c < c1; ++c) {
        const half16 h = vload16(c, row);
        const int8 b = (int8)(as_int((half2)(h.s0, h.s4)), as_int((half2)(h.s1, h.s5)),
                              as_int((half2)(h.s2, h.s6)), as_int((half2)(h.s3, h.s7)),
                              as_int((half2)(h.s8, h.sc)), as_int((half2)(h.s9, h.sd)),
                              as_int((half2)(h.sa, h.se)), as_int((half2)(h.sb, h.sf)));
        ushort8 av[PF_MB];
        PF_LOAD_A(av, at, K, M, row0, c)
        __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)
            acc[mb] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(av[mb]), b, acc[mb]);
    }
    __attribute__((opencl_unroll_hint)) for (int mb = 0; mb < PF_MB; ++mb)
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int r = row0 + mb * 8 + m;
            if (r < M)
                out[((size_t)ks * M + r) * N + rb * 16 + ln] = acc[mb][m];
        }
}

// ---------------------------------------------------------------------------
// Natural-order [M, K] -> P-order, one work-item per 16-chunk.
// ---------------------------------------------------------------------------
__kernel void pf_tile(const __global half* src, const int K, const int M, __global half* dst) {
    const int c = get_global_id(0);
    const int r = get_global_id(1);
    if (c >= K / 16 || r >= M)
        return;
    const ushort16 x = vload16(c, (const __global ushort*)src + (size_t)r * K);
    vstore16((ushort16)(x.s0, x.s4, x.s1, x.s5, x.s2, x.s6, x.s3, x.s7,
                        x.s8, x.sc, x.s9, x.sd, x.sa, x.se, x.sb, x.sf), c, (__global ushort*)dst + (size_t)r * K);
}

// ---------------------------------------------------------------------------
// MoE gather: row r = src row pair_of_row[r] / TOPK_ (zeros for padding) of a
// [*, K] f16 activation; only hdr[0] tiles exist. Sub-group i of row r copies
// 512-byte block i.
// ---------------------------------------------------------------------------
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void pf_gather(const __global uint* src, const int K, const __global int* hdr,
                        const __global int* pair_of_row, __global uint* dst) {
    const int r = get_group_id(1);
    if (r >= hdr[0] * PF_MT)
        return;
    const int p = pair_of_row[r];
    const size_t words = K / 2;
    const int i = get_sub_group_id() * 128;
    const uint8 v = p < 0 ? (uint8)(0) : intel_sub_group_block_read8(src + (size_t)(p / TOPK_) * words + i);
    intel_sub_group_block_write8(dst + r * words + i, v);
}

// ---------------------------------------------------------------------------
// One work-group per token: RMS(x) * w_in, P-order (the q|k|v GEMM input).
// ---------------------------------------------------------------------------
REQD __kernel void pf_attn_rms(const __global half* x, const __global half* w_in, __global half* at) {
    __local float tmp[NSG];
    const int t = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    x += (size_t)t * H_;
    float8 hv[HB_PER_LANE];
    float acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        hv[i] = b < HB ? vload_half8(b, x) : (float8)(0.0f);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        if (b < HB)
            pf_store8(hv[i] * inv * vload_half8(b, w_in), b, at + (size_t)t * H_);
    }
}

// ---------------------------------------------------------------------------
// One work-group per token, after o_proj:
//   h1 = h + RMS(o) * w_pa     (f32)
//   xd = RMS(h1) * w_pf        dense MLP input
//   xr = RMS(h1) * w_r         router input
//   x2 = RMS(h1) * w_pf2       MoE input, gathered per expert later
// (all three P-order f16)
// ---------------------------------------------------------------------------
REQD __kernel void pf_ffn_in(const __global float* o, const __global half* h,
                             const __global half* w_pa, const __global half* w_pf,
                             const __global half* w_r, const __global half* w_pf2,
                             __global float* h1_out, __global half* xd, __global half* xr,
                             __global half* x2) {
    __local float tmp[NSG];
    const int t = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    o += (size_t)t * H_;
    h += (size_t)t * H_;
    float8 hv[HB_PER_LANE];
    float acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        hv[i] = b < HB ? vload8(b, o) : (float8)(0.0f);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv_o = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    acc2 = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        if (b < HB)
            hv[i] = vload_half8(b, h) + hv[i] * inv_o * vload_half8(b, w_pa);
        acc2 += dot(hv[i].lo, hv[i].lo) + dot(hv[i].hi, hv[i].hi);
    }
    const float inv1 = rsqrt(wg_sum(acc2, tmp) / H_ + EPS_);
    const size_t ro = (size_t)t * H_;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HB_PER_LANE; ++i) {
        const int b = sg * SG + ln + LWS * i;
        if (b < HB) {
            vstore8(hv[i], b, h1_out + ro);
            const float8 n = hv[i] * inv1;
            pf_store8(n * vload_half8(b, w_pf), b, xd + ro);
            pf_store8(n * vload_half8(b, w_r), b, xr + ro);
            pf_store8(n * vload_half8(b, w_pf2), b, x2 + ro);
        }
    }
}

// One sub-group per token: sums the KS router partials (into slice 0; every
// lane reads back only what it wrote), then the top-k softmax.
REQD __kernel void pf_topk(__global float* logits, const int T, const int KS, __global int* tk_idx,
                           __global float* tk_w) {
    const int t = get_group_id(0) * NSG + get_sub_group_id();
    if (t >= T)
        return;
    __global float* l = logits + (size_t)t * E_;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i) {
        const int n = get_sub_group_local_id() + SG * i;
        float v = l[n];
        for (int ks = 1; ks < KS; ++ks)
            v += l[(size_t)ks * T * E_ + n];
        l[n] = v;
    }
    topk_softmax(l, 0, true, tk_idx + t * TOPK_, tk_w + t * TOPK_);
}

// ---------------------------------------------------------------------------
// Buckets the T*TOPK_ (token, slot) pairs per expert into PF_MT-row tiles.
// hdr[0] = tile count; tile_exp[tile] = expert; pair_of_row[row] = pair or -1.
// ---------------------------------------------------------------------------
__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void pf_moe_route(const __global int* tk_idx, const int T, __global int* hdr, __global int* tile_exp,
                           __global int* pair_of_row) {
    __local int cnt[E_], base[E_], fill[E_];
    const int lid = get_local_id(0);
    const int P = T * TOPK_;
    for (int e = lid; e < E_; e += 1024) {
        cnt[e] = 0;
        fill[e] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int p = lid; p < P; p += 1024)
        atomic_inc(&cnt[tk_idx[p]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int acc = 0;
        for (int e = 0; e < E_; ++e) {
            base[e] = acc;
            acc += (cnt[e] + PF_MT - 1) / PF_MT;
        }
        hdr[0] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int e = lid; e < E_; e += 1024) {
        const int nt = (cnt[e] + PF_MT - 1) / PF_MT;
        for (int i = 0; i < nt; ++i)
            tile_exp[base[e] + i] = e;
        for (int r = cnt[e]; r < nt * PF_MT; ++r)
            pair_of_row[base[e] * PF_MT + r] = -1;
    }
    for (int p = lid; p < P; p += 1024) {
        const int e = tk_idx[p];
        pair_of_row[base[e] * PF_MT + atomic_inc(&fill[e])] = p;
    }
}

// ---------------------------------------------------------------------------
// One work-group per token: the ffn_out tail with e = sum of the weighted
// expert rows ys[t * TOPK_ + slot].
// ---------------------------------------------------------------------------
REQD __kernel void pf_ffn_out(const __global float* h1, const __global float* m, const __global half* ys,
                              const __global half* w1, const __global half* w2, const __global half* w_post,
                              const __global half* layer_scalar, __global half* out) {
    __local float tmp[NSG];
    const int t = get_group_id(0);
    const int lid = get_local_id(0);
    h1 += (size_t)t * H_;
    m += (size_t)t * H_;
    ys += (size_t)t * TOPK_ * H_;
    out += (size_t)t * H_;
    float mv[HPT], ev[HPT];
    float sm = 0.0f, se = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        mv[i] = m[k];
        float e = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int s = 0; s < TOPK_; ++s)
            e += vload_half(s * H_ + k, ys);
        ev[i] = e;
        sm += mv[i] * mv[i];
        se += ev[i] * ev[i];
    }
    const float inv_m = rsqrt(wg_sum(sm, tmp) / H_ + EPS_);
    const float inv_e = rsqrt(wg_sum(se, tmp) / H_ + EPS_);
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        mv[i] = mv[i] * inv_m * vload_half(k, w1) + ev[i] * inv_e * vload_half(k, w2);
        ss += mv[i] * mv[i];
    }
    const float inv_s = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
    const float ls = vload_half(0, layer_scalar);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT; ++i) {
        const int k = lid + i * LWS;
        vstore_half((h1[k] + mv[i] * inv_s * vload_half(k, w_post)) * ls, k, out);
    }
}
)CLC";



// Whole-model Gemma-4 MegaKernel (kModel, see gemma4POCParams.h).
//
// Decode (one token) is ONE persistent launch, mk_decode: NWG co-resident
// work-groups walk all 30 decoder layers and lm_head, separated by grid-wide
// barriers (atomic counters in device memory). Per layer:
//   A  x = ffn_out(prev layer); q|k|v = Wqkv . (RMS(x) * w_in)
//   B  q/k norm + RoPE, KV-cache append, split-K flash decoding
//   C  o = o_proj(attention)
//   D  h1 = x + RMS(o) w_pa; router logits, dense gate/up, then the MoE
//      gate/up of the routed experts
//   E  gelu(gate) * up, MoE down (weighted by the routing weights), dense down
// and finally G: x = ffn_out(layer 29), lm_head (u8 per-row) + logit soft-cap.
//
// Every int4 GEMV phase is a flat stream of repacked records (repack_row_blocks):
// the records of all matrices of a phase are concatenated and each sub-group of
// the grid takes an equal contiguous slice, so all NWG * 16 sub-groups stream
// the same number of bytes. A slice may start or end inside a row block, so
// row sums are combined with fixed-point atomics. Accumulators are zeroed two
// phases after their last reader (see the zeroing notes in mk_decode).
//
// Prompt calls (several tokens) instead run the batched kernels of the per-layer
// runtimes plus the helpers at the end of this file.
//
// Appended to kFfnSource; built with its -D flags plus NL_, NWG, V_, RMAX_.
static const char* kModelSource = R"CLC(
#ifndef NWG
#define NWG 80
#endif
#define NSGT (NWG * NSG)
#define ATT_DMAX 512
#define NPH 5                          // grid barriers per layer
#define ATT_QD 512                     // q elements one sub-group keeps in registers (GPS * D)
#ifndef XMX_QK_UNROLL
#define XMX_QK_UNROLL 4                // attn_unit_xmx: Q K^T k-chunks per unrolled step
#endif
#ifndef ATT_KB
#define ATT_KB 1                       // keys whose K / V loads are in flight together
#endif
#ifndef SPIN_PAUSE
#define SPIN_PAUSE 8
#endif
#ifndef SPIN_MAX
#define SPIN_MAX (1 << 24)
#endif
#ifndef PF_RECS
#define PF_RECS 4                      // records per sub-group prefetched across a grid barrier
#endif
#ifndef OPROJ_PF_MAX
#define OPROJ_PF_MAX 12000             // records of o_proj prefetched during attention (~6.9 MB)
#endif
#ifndef ROUTE_SHARE
#define ROUTE_SHARE 8                  // phase D: routing sub-group's dense share, in 16ths
#endif
#if 2 * TOPK_ != SG
#error phase D posts one MoE gate/up job per lane of the routing sub-group
#endif

extern void __builtin_IB_eu_thread_pause(uint);

// Host mirror: LayerTabH / ModelTabH below (same member order).
typedef struct {
    const __global half* w_in;
    const __global uint* qkv;
    const __global half* gq;
    const __global half* gk;
    const __global float* invf;
    const __global uint* oproj;
    const __global half* w_pa;
    const __global half* w_pf;
    const __global half* w_r;
    const __global half* w_pf2;
    const __global uint* gate;
    const __global uint* up;
    const __global uint* down;
    const __global half* rw;
    const __global uint* egate;
    const __global uint* eup;
    const __global uint* edown;
    const __global half* w_pf1;
    const __global half* w_pf2p;
    const __global half* w_post;
    const __global half* ls;
    __global half* kc;                 // [HK][cap][D]
    __global half* vc;
    int R, HQ, HK, D, RN, q_off, k_off, v_off, half_c, pos_f16, sliding, pad;
} LayerTab;

typedef struct {
    __global half* xbuf;               // [H] residual entering the current layer (f16, as the graph)
    __global long* raw;                // [RMAX_] q|k|v projections (accumulators: fixed point, see atomic_addfx)
    __global half* att;                // [8192] attention output
    __global long* o;                  // [H] o_proj
    __global float* h1;                // [H]
    __global long* gu;                 // [2 * I_] dense gate | up
    __global long* lg;                 // [E_] router logits
    __global long* egu;                // [TOPK_][2][IE_] expert gate | up
    __global long* m;                  // [H] dense down
    __global long* e;                  // [H] MoE down
    __global float* part;              // attention partials
    __global int* sync;                // grid barrier counters
    __global int* tick;                // [NL_ * 8] attention combine tickets
    __global int* err;
    const __global half* w_final;
    const __global uint* lm;           // u8 lm_head records (16 rows x 64 bytes)
    const __global half* lm_s;         // [V_]
    const __global uchar* lm_z;        // [V_]
    float cap_div, cap_mul, attn_scale, window;
    float causal_off, p_mul, p_add, pad0;
    int pos_stride, am_stride, kv_cap, pad1;
    __global ulong* prof;              // MK_PROF: [barrier][NWG][arrive, release, ...]
    __global float* kmask;             // [kv_cap] additive mask of each key, filled by mk_decode
    __global half* qs;                 // [NL_][16][ATT_DMAX] roped f16 queries (attn_unit_xmx)
} ModelTab;

extern ulong __builtin_IB_read_cycle_counter(void);
#ifdef MK_PROF
#ifdef MK_PROF_TOTAL  // kernel start / end only
#define PROF_MARK(mt, bar, which)                                                                   \
    if (get_local_id(0) == 0 && (bar) == NL_ * NPH + 1)                                             \
        (mt)->prof[((size_t)(bar) * NWG + get_group_id(0)) * 5 + (which)] = __builtin_IB_read_cycle_counter();
#define PROF_DBG(mt, layer, u, point)
#else
#define PROF_MARK(mt, bar, which)                                                                   \
    if (get_local_id(0) == 0)                                                                       \
        (mt)->prof[((size_t)(bar) * NWG + get_group_id(0)) * 5 + (which)] = __builtin_IB_read_cycle_counter();
#define PROF_DBG(mt, layer, u, point)                                                               \
    if (get_local_id(0) == 0 && (u) == 0)                                                           \
        (mt)->prof[(size_t)(NL_ * NPH + 2) * NWG * 5 + (layer) * 16 + (point)] = __builtin_IB_read_cycle_counter();
#endif
#else
#define PROF_MARK(mt, bar, which)
#define PROF_DBG(mt, layer, u, point)
#endif

// ---------------------------------------------------------------------------
// Grid barrier: every work-group increments the counter once; waiters spin
// until it reaches epoch * NWG. Counters are never reset, the epoch grows by
// one per launch. A bounded spin turns a co-residency failure into garbage
// output plus an error flag instead of a GPU hang.
// ---------------------------------------------------------------------------
inline void grid_arrive(__global int* c) {
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
        atomic_fetch_add_explicit((volatile __global atomic_int*)c, 1, memory_order_release, memory_scope_device);
}

inline void grid_wait(__global int* c, int target, __global int* err) {
    if (get_local_id(0) == 0) {
        int n = 0;
        while (atomic_load_explicit((volatile __global atomic_int*)c, memory_order_relaxed, memory_scope_device) <
               target) {
            __builtin_IB_eu_thread_pause(SPIN_PAUSE);
            if (++n > SPIN_MAX) {
                *err = 1;
                break;
            }
        }
        atomic_load_explicit((volatile __global atomic_int*)c, memory_order_acquire, memory_scope_device);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}

// grid_wait for a single sub-group (no work-group barrier).
inline void sg_wait(__global int* c, int target, __global int* err) {
    if (get_sub_group_local_id() == 0) {
        int n = 0;
        while (atomic_load_explicit((volatile __global atomic_int*)c, memory_order_relaxed, memory_scope_device) <
               target) {
            __builtin_IB_eu_thread_pause(SPIN_PAUSE);
            if (++n > SPIN_MAX) {
                *err = 1;
                break;
            }
        }
    }
    sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
    atomic_load_explicit((volatile __global atomic_int*)c, memory_order_acquire, memory_scope_device);
}

// Split-K sums are accumulated as Q32.32 fixed point with 64-bit integer
// atomics: integer addition is associative, so the result is the exact sum of
// the (float) partials whatever order the sub-groups arrive in - unlike float
// atomics, which made every run differ slightly and flipped near-tied MoE
// experts. |sum| must stay below 2^31.
#define FX_ONE 4294967296.0f
inline void atomic_addfx(__global long* p, float v) {
    atomic_fetch_add_explicit((volatile __global atomic_long*)p, convert_long_rte(v * FX_ONE), memory_order_relaxed,
                              memory_scope_device);
}
inline float fx2f(long v) {
    return (float)v * (1.0f / FX_ONE);
}
inline float8 fx2f8(long8 v) {
    return convert_float8(v) * (1.0f / FX_ONE);
}

// 0.5 x (1 + tanh(u)) written as x / (1 + exp(-2u)): the same function, but
// OpenCL's tanh made phase E's prologue ~2x slower.
inline float8 gelu_tanh8(float8 x) {
    return x / (1.0f + exp(-1.5957691216057308f * (x + 0.044715f * x * x * x)));
}

// topk_softmax with the result in SLM; called by one sub-group.
inline void topk_local(const __global long* logits, __local int* tk_idx, __local float* tk_w) {
    const int ln = get_sub_group_local_id();
    float v[E_ / SG];
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i) {
        const float l = fx2f(logits[ln + SG * i]);
        v[i] = isnan(l) ? -INFINITY : l;
    }
    float top = 0.0f, wsum = 0.0f, mine = 0.0f;
    int mine_idx = 0;
    __attribute__((opencl_unroll_hint)) for (int t = 0; t < TOPK_; ++t) {
        float lm = v[0];
        int li = ln;
        __attribute__((opencl_unroll_hint)) for (int i = 1; i < E_ / SG; ++i) {
            if (v[i] > lm) {
                lm = v[i];
                li = ln + SG * i;
            }
        }
        const float mxv = sub_group_reduce_max(lm);
        const int idx = sub_group_reduce_min(lm == mxv ? li : 0x7fffffff);
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < E_ / SG; ++i)
            if (ln + SG * i == idx)
                v[i] = -INFINITY;
        if (t == 0)
            top = mxv;
        const float e = exp(mxv - top);
        wsum += e;
        if (ln == t) {
            mine = e;
            mine_idx = idx;
        }
    }
    if (ln < TOPK_) {
        tk_idx[ln] = mine_idx;
        tk_w[ln] = mine / wsum;
    }
}

inline void zero_grid(__global long* p, int n) {
    for (int i = get_group_id(0) * LWS + get_local_id(0); i < n; i += NWG * LWS)
        p[i] = 0;
}

inline void prefetch_recs(const __global uint* rec, int n) {
    for (int i = 0; i < n; ++i) {
        intel_sub_group_block_prefetch_ui8(rec + i * REC_DW);
        intel_sub_group_block_prefetch_ui(rec + i * REC_DW + 128);
    }
}

// Pulls a small vector (an RMS gamma) into L2 ahead of the prologue that reads
// it: the per-layer vectors are evicted by the weight stream, and their DRAM
// latency otherwise lands on the critical path of every phase prologue.
inline void prefetch_bytes(const __global void* p, int bytes) {
    for (int o = 0; o < bytes; o += 512)
        intel_sub_group_block_prefetch_ui8((const __global uint*)((const __global uchar*)p + o));
}

// The contiguous record slice [r0, r1) of `total` records owned by sub-group gs of ns.
inline int slice_lo(int total, int gs, int ns) {
    return (int)(((long)total * gs) / ns);
}

// ---------------------------------------------------------------------------
// Flat record streaming over a job list (int4 GEMV phases)
// ---------------------------------------------------------------------------
typedef struct {
    const __global uint* m;            // records of the job's first row block
    __global long* out;                // row r of the job -> out[r]
    int G;                             // groups per row block
    int nrec;                          // records of the job
    int xq;                            // SLM word offset of the quantised activation
    int K;                             // its length
    int gx;                            // activation group of the job's group 0
    float wmul;
} Job;

// One group of a record against the activation group g.
#define GROUP_DOT(wv, meta, g, xh, xl, xa, xs, acc)                                                     \
    {                                                                                                 \
        const uint xhv_ = intel_sub_group_block_read(xh + (g) * 16);                                  \
        const uint xlv_ = intel_sub_group_block_read(xl + (g) * 16);                                  \
        int ih = 0, il = 0;                                                                           \
        DOT8(wv, xhv_, xlv_, ih, il)                                                                  \
        const float sc_ = (float)as_half2(meta).s0;                                                   \
        const float zz_ = (float)((meta) >> 16);                                                      \
        acc = fma(sc_, fma(xa[g], fma((float)il, 1.0f / XL_RES, (float)ih), -zz_ * xs[g]), acc);     \
    }

#ifndef SEG_BATCH
#define SEG_BATCH 4
#endif
// acc over groups [g0, g0 + ng) of one row block; rec = record of group g0.
// Records are loaded SEG_BATCH at a time, every load before the first use.
inline float seg_dot(const __global uint* rec, const int g0, const int ng, const __local uint* xq, const int K) {
    const __local uint* xh = xq;
    const __local uint* xl = xq + K / 4;
    const __local float* xa = (const __local float*)(xq + K / 2);
    const __local float* xs = xa + K / QG;
    float acc = 0.0f;
    int gi = 0;
    for (; gi + SEG_BATCH <= ng; gi += SEG_BATCH) {
        uint8 wv[SEG_BATCH];
        uint mv[SEG_BATCH];
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < SEG_BATCH; ++b) {
            wv[b] = intel_sub_group_block_read8(rec + (gi + b) * REC_DW);
            mv[b] = intel_sub_group_block_read(rec + (gi + b) * REC_DW + 128);
        }
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < SEG_BATCH; ++b)
            GROUP_DOT(wv[b], mv[b], g0 + gi + b, xh, xl, xa, xs, acc)
    }
    for (; gi < ng; ++gi) {
        const uint8 wv = intel_sub_group_block_read8(rec + gi * REC_DW);
        const uint m_ = intel_sub_group_block_read(rec + gi * REC_DW + 128);
        GROUP_DOT(wv, m_, g0 + gi, xh, xl, xa, xs, acc)
    }
    return acc;
}

// Records [r, r1) of the concatenated jobs.
inline void run_jobs_range(const __local Job* jobs, const int njobs, int r, const int r1, const __local uint* slm) {
    int j = 0, base = 0;
    while (j < njobs && r >= base + jobs[j].nrec) {
        base += jobs[j].nrec;
        ++j;
    }
    const int ln = get_sub_group_local_id();
    while (r < r1) {
        const Job jb = jobs[j];
        const int lr = r - base;
        const int rb = lr / jb.G;
        const int g0 = lr - rb * jb.G;
        const int end = min(r1 - base, (rb + 1) * jb.G);
        const int ng = end - lr;
        const float acc = seg_dot(jb.m + (size_t)lr * REC_DW, jb.gx + g0, ng, slm + jb.xq, jb.K);
        atomic_addfx(jb.out + rb * SG + ln, acc * jb.wmul);
        r += ng;
        if (r - base >= jb.nrec) {
            base += jb.nrec;
            ++j;
        }
    }
}

// Sub-group gs of ns streams its slice of the concatenated jobs.
inline void run_jobs(const __local Job* jobs, const int njobs, const int gs, const int ns, const __local uint* slm) {
    int total = 0;
    for (int j = 0; j < njobs; ++j)
        total += jobs[j].nrec;
    run_jobs_range(jobs, njobs, slice_lo(total, gs, ns), slice_lo(total, gs + 1, ns), slm);
}

#ifndef POOL_PCT
#define POOL_PCT 25                    // run_jobs_pool share of A, C, D (percent)
#endif
#ifndef POOL_PCT_E
#define POOL_PCT_E 25                  // ... and of E
#endif
#ifndef POOL_CH
#define POOL_CH 4
#endif
// Phase E builds the dense activation for its pooled tail, so that tail must
// lie inside the dense down job (the last one).
#if POOL_PCT_E * (TOPK_ * (H_ / SG) * (IE_ / QG) + (H_ / SG) * (I_ / QG)) > 100 * (H_ / SG) * (I_ / QG)
#error POOL_PCT_E too large
#endif
// Records of a run_jobs_pool phase that are sliced statically; the rest form the pool.
inline int pool_static(int total, int pct) {
    return total - total * pct / 100 / POOL_CH * POOL_CH;
}

// run_jobs with the last pct % of the records handed out dynamically in
// POOL_CH-record chunks, so sub-groups that stream faster take more of the
// tail. ctr is this phase's claim counter: every sub-group claims until one
// claim fails, so a launch adds exactly npool + ns and the epoch gives its base.
// Chunk boundaries are fixed, so the sums do not depend on who claims what.
inline void run_jobs_pool(const __local Job* jobs, const int njobs, const int gs, const int ns, const __local uint* slm,
                          const int pct, __global int* ctr, const int epoch) {
    int total = 0;
    for (int j = 0; j < njobs; ++j)
        total += jobs[j].nrec;
    const int S = pool_static(total, pct);
    const int npool = (total - S) / POOL_CH;
    run_jobs_range(jobs, njobs, slice_lo(S, gs, ns), slice_lo(S, gs + 1, ns), slm);
    // uint: the counter wraps after ~2^32 / (npool + ns) launches.
    const uint base = (uint)(epoch - 1) * (uint)(npool + ns);
    for (;;) {
        int c = 0;
        if (get_sub_group_local_id() == 0)
            c = (int)((uint)atomic_fetch_add_explicit((volatile __global atomic_int*)ctr, 1, memory_order_relaxed,
                                                      memory_scope_device) - base);
        c = sub_group_broadcast(c, 0);
        if (c >= npool)
            break;
        run_jobs_range(jobs, njobs, S + c * POOL_CH, S + (c + 1) * POOL_CH, slm);
    }
}

// First record of sub-group gs's slice of a single-matrix phase.
inline void prefetch_single(const __global uint* m, int total, int gs, int ns) {
    const int r0 = slice_lo(total, gs, ns), r1 = slice_lo(total, gs + 1, ns);
    prefetch_recs(m + (size_t)r0 * REC_DW, min(PF_RECS, r1 - r0));
}

// ---------------------------------------------------------------------------
// Element-wise prologues. Work-item lid owns elements lid + LWS * i.
// ---------------------------------------------------------------------------
#define HPT_ (H_ / LWS)

// Two wg_sum results for one pair of barriers.
inline float2 wg_sum2(float a, float b, __local float* tmp) {
    a = sub_group_reduce_add(a);
    b = sub_group_reduce_add(b);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0) {
        tmp[get_sub_group_id()] = a;
        tmp[NSG + get_sub_group_id()] = b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return (float2)(sub_group_reduce_add(tmp[get_sub_group_local_id()]),
                    sub_group_reduce_add(tmp[NSG + get_sub_group_local_id()]));
}

// Residual leaving layer P (ffn_out of the per-layer kernels), rounded to f16
// as the graph stores it; ls is P's layer scalar. Also loads gn = gnext[k] for
// the caller's next RMS: every global load is issued before the first
// reduction so the latencies overlap instead of adding up.
inline void ffn_out_vals(const __global LayerTab* P, const __global ModelTab* mt, const __global half* gnext,
                         const float ls, float xv[HPT_], float gn[HPT_], __local float* tmp) {
    const int lid = get_local_id(0);
    const __global long* m = mt->m;
    const __global long* e = mt->e;
    const __global float* h1 = mt->h1;
    const __global half* w_pf1 = P->w_pf1;
    const __global half* w_pf2p = P->w_pf2p;
    const __global half* w_post = P->w_post;
    float mv[HPT_], ev[HPT_], g1[HPT_], g2[HPT_], gp[HPT_], hv[HPT_];
    float sm = 0.0f, se = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
        const int k = lid + i * LWS;
        mv[i] = fx2f(m[k]);
        ev[i] = fx2f(e[k]);
        hv[i] = h1[k];
        g1[i] = vload_half(k, w_pf1);
        g2[i] = vload_half(k, w_pf2p);
        gp[i] = vload_half(k, w_post);
        gn[i] = vload_half(k, gnext);
    }
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
        sm += mv[i] * mv[i];
        se += ev[i] * ev[i];
    }
    const float2 s2 = wg_sum2(sm, se, tmp);
    const float inv_m = rsqrt(s2.x / H_ + EPS_);
    const float inv_e = rsqrt(s2.y / H_ + EPS_);
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
        mv[i] = mv[i] * inv_m * g1[i] + ev[i] * inv_e * g2[i];
        ss += mv[i] * mv[i];
    }
    const float inv_s = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
        xv[i] = (float)convert_half((hv[i] + mv[i] * inv_s * gp[i]) * ls);
}

// dst[k] = v[k] * RMS(v) * g[k] into SLM.
inline void rms_to_slm_v(const float v[HPT_], const float g[HPT_], __local float* dst, __local float* tmp) {
    const int lid = get_local_id(0);
    float ss = 0.0f;
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
        ss += v[i] * v[i];
    const float inv = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
        dst[lid + i * LWS] = v[i] * inv * g[i];
}

inline void rms_to_slm(const float v[HPT_], const __global half* gamma, __local float* dst, __local float* tmp) {
    float g[HPT_];
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
        g[i] = vload_half(get_local_id(0) + i * LWS, gamma);
    rms_to_slm_v(v, g, dst, tmp);
}

// Quantises K floats from SLM into the GEMV activation layout (whole WG).
inline void quant_slm(const __local float* src, const int K, __local uint* xq) {
    for (int c0 = get_sub_group_id() * SG; c0 < K / 8; c0 += NSG * SG) {
        const int b = c0 + get_sub_group_local_id();
        const bool ok = b < K / 8;
        const QB q = quant_block(ok ? vload8(b, src) : (float8)(0.0f), ok);
        if (ok)
            store_qb_l(q, b, K, xq);
    }
}

// ---------------------------------------------------------------------------
// Phase B: attention for one (kv head, key split) unit.
// Each sub-group keeps GPS = ATT_QD / D query heads in registers (strided
// layout: element d = j * 16 + lane) and walks every NKL-th key of the split.
// ---------------------------------------------------------------------------
// Key range [j0, j1) of split `split` out of NS over the visible keys [jlo, jhi].
inline int2 split_keys(const int jlo, const int jhi, const int NS, const int split) {
    const int chunk = (jhi - jlo + NS) / NS;
    const int j0 = jlo + split * chunk;
    return (int2)(j0, min(jhi + 1, j0 + chunk));
}

// Head vector of one lane (element d = j * 16 + lane), RMS-normalised with
// gamma (or without when g is null), then rotate-half RoPE over d < RN. The
// partner of element j is j +- RN / 32 of the same lane. All loads are issued
// before the first use.
// Loads use sub-group block reads: one uniform address returns exactly the
// lane-strided layout, instead of a 64-bit address per lane and element.
#define HEAD_NORM_ROPE(OUT, OFF, NJ, RNL, r, g, cs, sn)                                                    \
    {                                                                                                     \
        float ss_ = 0.0f;                                                                                 \
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < (NJ) / 8; ++c) {                          \
            const float8 v_ = fx2f8(as_long8(intel_sub_group_block_read_ul8((const __global ulong*)(r) + c * 128))); \
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i) {                             \
                OUT[(OFF) + c * 8 + i] = v_[i];                                                           \
                ss_ = fma(v_[i], v_[i], ss_);                                                             \
            }                                                                                             \
        }                                                                                                 \
        const float inv_ = rsqrt(sub_group_reduce_add(ss_) / (NJ * SG) + EPS_);                          \
        if (g) {                                                                                          \
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < (NJ) / 8; ++c) {                      \
                const half8 g_ = as_half8(intel_sub_group_block_read_us8((const __global ushort*)(g) + c * 128)); \
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)                           \
                    OUT[(OFF) + c * 8 + i] *= inv_ * (float)g_[i];                                        \
            }                                                                                             \
            const int hr_ = (RNL) / SG / 2;                                                               \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < hr_; ++j) {                           \
                const float a_ = OUT[(OFF) + j], b_ = OUT[(OFF) + j + hr_];                               \
                const int d_ = j * SG + ln, e_ = d_ + hr_ * SG;                                           \
                OUT[(OFF) + j] = a_ * cs[d_] - b_ * sn[d_];                                               \
                OUT[(OFF) + j + hr_] = b_ * cs[e_] + a_ * sn[e_];                                         \
            }                                                                                             \
        } else {                                                                                          \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < (NJ); ++j)                            \
                OUT[(OFF) + j] *= inv_;                                                                   \
        }                                                                                                 \
    }

// Cross-unit merge of the NS key splits of kv head kvh, called by every unit
// once its partials are in mt->part (PS = D + 16 floats per (head, split): the
// D accumulators, the max, the sum). The units wait for each other, then each
// merges a 1 / NS share of the G * D outputs. The counter grows by NS per
// launch; att_base is its value before this launch. tmp: 1088 floats, wt: G * NS.
inline void attn_merge(const __global ModelTab* mt, const int layer, const int kvh, const int split, const int NS,
                       const int G, const int D, __local float* tmp, __local float* wt, const int att_base) {
    const int lid = get_local_id(0);
    const int PS = D + 16;
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    __global int* t = mt->tick + layer * 8 + kvh;
    grid_arrive(t);
    grid_wait(t, att_base + NS, mt->err);
    const int chunk = (G * D + NS - 1) / NS;
    const int o0 = split * chunk, o1 = min(G * D, o0 + chunk);
    const int h0 = o0 / D, h1 = (o1 - 1) / D;
    // Per (head, split) weights exp(m - M) / L: gather m, l once, reduce per head.
    const int nh = h1 - h0 + 1;
    __local float* mm = tmp;
    __local float* ll = tmp + 8 * 64;
    __local float* hml = tmp + 16 * 64;
    for (int i = lid; i < nh * NS; i += LWS) {
        const __global float* pp = mt->part + ((size_t)(kvh * G + h0 + i / NS) * NS + i % NS) * PS + D;
        mm[i] = pp[0];
        ll[i] = pp[1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < nh) {
        float Mx = -INFINITY, Lsum = 0.0f;
        for (int p = 0; p < NS; ++p)
            Mx = fmax(Mx, mm[lid * NS + p]);
        for (int p = 0; p < NS; ++p)
            Lsum = fma(exp(mm[lid * NS + p] - Mx), ll[lid * NS + p], Lsum);
        hml[2 * lid] = Mx;
        hml[2 * lid + 1] = 1.0f / Lsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lid; i < nh * NS; i += LWS)
        wt[i] = exp(mm[i] - hml[2 * (i / NS)]) * hml[2 * (i / NS) + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int o = o0 + lid; o < o1; o += LWS) {
        const int hw = o / D, d = o - hw * D;
        const __global float* pb = mt->part + (size_t)(kvh * G + hw) * NS * PS + d;
        float a = 0.0f;
        __attribute__((opencl_unroll_hint(8))) for (int p = 0; p < NS; ++p)
            a = fma(wt[(hw - h0) * NS + p], pb[p * PS], a);
        mt->att[(kvh * G + hw) * D + d] = convert_half(a);
    }
}

// D, RN and GPS are literals at both call sites, so after inlining every
// register array is statically indexed. Each sub-group keeps GPS = 512 / D
// query heads and walks every NKL-th key of the split, KB = GPS keys per
// batch with all of their loads in flight together.
__attribute__((always_inline)) inline void attn_unit(const __global LayerTab* L, const __global ModelTab* mt, const int layer, const int u,
                      const int NS, const int past, const int jlo, const int jhi, const float posf,
                      const __global float* mk, __local float* slm, const int att_base, const int D,
                      const int RN) {
    const int GPS = ATT_QD / D, KB = ATT_KB;
    const int nj = D / SG;
    const int HQ = L->HQ, HK = L->HK;
    const int G = HQ / HK;
    const int QSPLIT = G / GPS;
    const int NKL = NSG / QSPLIT;
    const int kvh = u / NS, split = u % NS;
    const int sg = get_sub_group_id(), ln = get_sub_group_local_id(), lid = get_local_id(0);
    const int qset = sg % QSPLIT, kl = sg / QSPLIT;
    const int cap = mt->kv_cap;
    const int half_c = L->half_c;
    const __global float* invf = L->invf;
    const __global half* gq = L->gq;
    const __global half* gk = L->gk;
    const __global long* raw = mt->raw;
    const int q_off = L->q_off, k_off = L->k_off, v_off = L->v_off;
    __global half* kc = L->kc + (size_t)kvh * cap * D;
    __global half* vc = L->vc + (size_t)kvh * cap * D;
    __local float* slm_acc = slm;                          // [NSG][ATT_QD]
    __local float* msg = slm + NSG * ATT_QD;               // [NSG][2]
    __local float* lsm = msg + 2 * NSG;                    // [NSG][2]
    __local float* wt = lsm + 2 * NSG;                     // [G][NS]
    __local half* kvcur = (__local half*)(wt + 8 * 64);    // [2][D]
    __local int* flag = (__local int*)(kvcur + 2 * ATT_DMAX);
    PROF_DBG(mt, layer, u, 0)

    const int2 jr = split_keys(jlo, jhi, NS, split);
    const int j0 = jr.x, j1 = jr.y;
#ifdef ATT_STUB
    return;
#endif

    // RoPE tables (one sin / cos per work-item), read per lane from SLM.
    __local float* cs = (__local float*)(flag + 4);
    __local float* sn = cs + ATT_DMAX;
    for (int i = lid; i < RN; i += LWS) {
        const float f = posf * invf[i % half_c];
        cs[i] = (float)convert_half_rte(cos(f));
        sn[i] = (float)convert_half_rte(sin(f));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#if ATT_STOP == 1
    mt->att[lid] = cs[lid] + sn[lid];
    return;
#endif
    // The current token's k / v: appended by the split that owns it, which
    // also reads them back from SLM instead of the cache.
    const bool own_cur = j1 == jhi + 1 && past <= jhi;
    if (own_cur && sg < 2) {
        float hv[ATT_DMAX / SG];
        const __global long* hr = raw + (sg == 0 ? k_off : v_off) + kvh * D;
        const __global half* hg = sg == 0 ? gk : (const __global half*)0;
        HEAD_NORM_ROPE(hv, 0, nj, RN, hr, hg, cs, sn)
        __global ushort* dst = (__global ushort*)((sg == 0 ? kc : vc) + (size_t)past * D);
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
            ushort8 h;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                h[i] = as_ushort(convert_half(hv[c * 8 + i]));
            intel_sub_group_block_write_us8(dst + c * 128, h);
            intel_sub_group_block_write_us8((__local ushort*)kvcur + sg * ATT_DMAX + c * 128, h);
        }
    }
#if ATT_STOP == 2
    return;
#endif
    float qv[ATT_QD / SG], acc[ATT_QD / SG];
    float mx[2], ls[2];
    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
        const int qh = kvh * G + qset * GPS + hh;
        HEAD_NORM_ROPE(qv, hh * nj, nj, RN, raw + q_off + qh * D, gq, cs, sn)
        // q is f16 in the graph (RoPE output); the rounding matters for the
        // peaked full-attention heads.
        const float qs = mt->attn_scale;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
            qv[hh * nj + j] = (float)convert_half(qv[hh * nj + j]) * qs;
        mx[hh] = -INFINITY;
        ls[hh] = 0.0f;
    }
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < ATT_QD / SG; ++i)
        acc[i] = 0.0f;
#ifdef ATT_NO_SLM_CUR
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
#else
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    PROF_DBG(mt, layer, u, 3)
#if ATT_STOP == 3
    { float z = 0; for (int i = 0; i < 32; ++i) z += qv[i]; mt->att[lid] = z; return; }
#endif

    for (int t0 = j0 + kl; t0 < j1; t0 += NKL * KB) {
#ifdef ATT_PFD
        {
            const int jp = t0 + ATT_PFD * NKL * KB;
            if (jp < j1)
                __attribute__((opencl_unroll_hint)) for (int c = 0; c < D / 256; ++c) {
                    intel_sub_group_block_prefetch_ui8((const __global uint*)(kc + (size_t)jp * D) + c * 128);
                    intel_sub_group_block_prefetch_ui8((const __global uint*)(vc + (size_t)jp * D) + c * 128);
                }
        }
#endif
        half8 kk[2][ATT_DMAX / SG / 8], vv[2][ATT_DMAX / SG / 8];
        float madd[2];
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < KB; ++b) {
            const int j = t0 + b * NKL;
            madd[b] = -INFINITY;
            if (j < j1) {
                madd[b] = mk[j];
#ifdef ATT_NO_SLM_CUR
                const bool cur = false;
#else
                const bool cur = j == past;
#endif
                const __global ushort* kp = (const __global ushort*)(kc + (size_t)j * D);
                const __global ushort* vp = (const __global ushort*)(vc + (size_t)j * D);
                const __local ushort* kl_ = (const __local ushort*)kvcur;
                __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                    kk[b][c] = as_half8(cur ? intel_sub_group_block_read_us8(kl_ + c * 128)
                                            : intel_sub_group_block_read_us8(kp + c * 128));
                    vv[b][c] = as_half8(cur ? intel_sub_group_block_read_us8(kl_ + ATT_DMAX + c * 128)
                                            : intel_sub_group_block_read_us8(vp + c * 128));
                }
            }
        }
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < KB; ++b) {
            if (madd[b] < -1e4f)
                continue;
            float s[2] = {0.0f, 0.0f};
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c)
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh)
                        s[hh] = fma(qv[hh * nj + c * 8 + i], (float)kk[b][c][i], s[hh]);
            float pr[2], cr[2];
            __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
                const float sc = sub_group_reduce_add(s[hh]) + madd[b];
                const float mn = fmax(mx[hh], sc);
                cr[hh] = exp(mx[hh] - mn);
                pr[hh] = exp(sc - mn);
                ls[hh] = fma(ls[hh], cr[hh], pr[hh]);
                mx[hh] = mn;
            }
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c)
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
                        const int k = hh * nj + c * 8 + i;
                        acc[k] = fma(acc[k], cr[hh], pr[hh] * (float)vv[b][c][i]);
                    }
        }
    }
    PROF_DBG(mt, layer, u, 4)
#if ATT_STOP == 4
    { float z = mx[0] + ls[0]; for (int i = 0; i < 32; ++i) z += acc[i]; mt->att[lid] = z; return; }
#endif

    // Work-group combine over the NKL key lanes of each head: rescale to the
    // common maximum, then a tree through SLM.
    if (ln == 0) {
        __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh)
            msg[sg * 2 + hh] = mx[hh];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float M[2];
    __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
        M[hh] = -INFINITY;
        for (int k = 0; k < NKL; ++k)
            M[hh] = fmax(M[hh], msg[(k * QSPLIT + qset) * 2 + hh]);
        const float w = mx[hh] == -INFINITY ? 0.0f : exp(mx[hh] - M[hh]);
        ls[hh] *= w;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
            acc[hh * nj + j] *= w;
    }
    for (int half_ = NKL / 2; half_ >= 1; half_ /= 2) {
        if (kl >= half_ && kl < 2 * half_) {
            const int slot = (kl - half_) * QSPLIT + qset;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < ATT_QD / SG; ++i)
                slm_acc[slot * ATT_QD + i * SG + ln] = acc[i];
            if (ln == 0) {
                lsm[slot * 2] = ls[0];
                lsm[slot * 2 + 1] = ls[1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kl < half_) {
            const int slot = kl * QSPLIT + qset;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < ATT_QD / SG; ++i)
                acc[i] += slm_acc[slot * ATT_QD + i * SG + ln];
            ls[0] += lsm[slot * 2];
            ls[1] += lsm[slot * 2 + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int PS = D + 16;  // keeps partials 64-byte aligned
    // A single split is final already.
    if (NS == 1) {
        if (kl == 0) {
            __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
                const int qh = kvh * G + qset * GPS + hh;
                const float il = 1.0f / ls[hh];
                __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                    ushort8 h;
                    __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                        h[i] = as_ushort(convert_half(acc[hh * nj + c * 8 + i] * il));
                    intel_sub_group_block_write_us8((__global ushort*)(mt->att + qh * D) + c * 128, h);
                }
            }
        }
        PROF_DBG(mt, layer, u, 5)
        PROF_DBG(mt, layer, u, 6)
        return;
    }
    if (kl == 0) {
        __attribute__((opencl_unroll_hint)) for (int hh = 0; hh < GPS; ++hh) {
            const int qh = kvh * G + qset * GPS + hh;
            __global float* pp = mt->part + ((size_t)qh * NS + split) * PS;
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c)
                intel_sub_group_block_write8((__global uint*)(pp + c * 128), as_uint8((float8)(
                    acc[hh * nj + c * 8], acc[hh * nj + c * 8 + 1], acc[hh * nj + c * 8 + 2],
                    acc[hh * nj + c * 8 + 3], acc[hh * nj + c * 8 + 4], acc[hh * nj + c * 8 + 5],
                    acc[hh * nj + c * 8 + 6], acc[hh * nj + c * 8 + 7])));
            if (ln == 0) {
                pp[D] = M[hh];
                pp[D + 1] = ls[hh];
            }
        }
    }
    PROF_DBG(mt, layer, u, 5)
#ifdef ATT_SKIP_COMBINE
    return;
#endif
    attn_merge(mt, layer, kvh, split, NS, G, D, slm_acc, wt, att_base);
    PROF_DBG(mt, layer, u, 6)
}

// Separate (non-inlined) instantiations keep the attention's register
// pressure out of the main loop of mk_decode.
#ifdef ATT_NOINLINE
#define ATT_FN_ATTR __attribute__((noinline))
#else
#define ATT_FN_ATTR
#endif
// One head of raw (q, k or v; RMS with gamma g or none, RoPE as HEAD_NORM_ROPE)
// stored as f16 to dst. D is a literal at the call sites.
__attribute__((always_inline)) inline void head_to_f16(const __global long* r, const __global half* g,
                                                       const __local float* cs, const __local float* sn,
                                                       __global ushort* dst, const int D) {
    const int ln = get_sub_group_local_id();
    float hv[ATT_DMAX / SG];
    HEAD_NORM_ROPE(hv, 0, D / SG, D, r, g, cs, sn)
    __attribute__((opencl_unroll_hint)) for (int c = 0; c < D / SG / 8; ++c) {
        ushort8 h;
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
            h[i] = as_ushort(convert_half(hv[c * 8 + i]));
        intel_sub_group_block_write_us8(dst + c * 128, h);
    }
}

// XMX variant of attn_unit: the G <= 8 query heads of kv head kvh are the 8
// rows of DPAS tiles (as mk_pf_fa's 8 queries), 16 keys per chunk:
//   S[8 x 16] = Q K^T   Q tiles from mt->qs (f16, written here first), K^T by
//                       a transposed 2D block read of the cache
//   online softmax per row, P rounded to f16
//   O[8 x 128] += P V   V tiles via the VNNI-transforming read
// The scalar unit spends most of its time converting and multiplying K / V per
// query head; here a key is touched once per 128-column slice. D is split into
// DQ = D / 128 column slices: sub-group sg takes slice sg % DQ of the key chunks
// grp, grp + NGRP, ... (grp = sg / DQ, NGRP = 16 / DQ); the groups are then
// combined through SLM and the splits by attn_merge. Rows >= G read as zero
// and are never stored. D (256 or 512) is a runtime value: one shared body,
// since inlining a copy per D made the kernel spill.
__attribute__((always_inline)) inline void attn_unit_xmx(const __global LayerTab* L, const __global ModelTab* mt,
                                                         const int layer, const int u, const int NS, const int past,
                                                         const int jlo, const int jhi, const float posf,
                                                         const __global float* mk, __local float* slm,
                                                         const int att_base, const int D) {
    const int G = L->HQ / L->HK;
    const int kvh = u / NS, split = u % NS;
    const int sg = get_sub_group_id(), ln = get_sub_group_local_id(), lid = get_local_id(0);
    const int DQ = D / 128, NGRP = NSG / DQ;
    const int grp = sg / DQ, dq = sg % DQ;
    const int cap = mt->kv_cap;
    const __global long* raw = mt->raw;
    __global half* kc = L->kc + (size_t)kvh * cap * D;
    __global half* vc = L->vc + (size_t)kvh * cap * D;
    __global half* qb = mt->qs + (size_t)layer * 16 * ATT_DMAX + kvh * G * D;  // [G][D], own region per layer
    PROF_DBG(mt, layer, u, 0)
    const int2 jr = split_keys(jlo, jhi, NS, split);
    const int j0 = jr.x, j1 = jr.y;

    // RoPE tables, then the current token's k / v (owner split, sub-groups 0
    // and 1) and the G roped f16 query rows (sub-groups 2 .. G + 1).
    __local float* cs = slm;
    __local float* sn = slm + ATT_DMAX;
    for (int i = lid; i < D; i += LWS) {
        const float f = posf * L->invf[i % L->half_c];
        cs[i] = (float)convert_half_rte(cos(f));
        sn[i] = (float)convert_half_rte(sin(f));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const bool own_cur = j1 == jhi + 1 && past <= jhi;
    if ((own_cur && sg < 2) || (sg >= 2 && sg < 2 + G)) {
        const __global long* hr = raw + (sg == 0 ? L->k_off + kvh * D
                                         : sg == 1 ? L->v_off + kvh * D
                                                   : L->q_off + (kvh * G + sg - 2) * D);
        const __global half* hg = sg == 0 ? L->gk : sg == 1 ? (const __global half*)0 : L->gq;
        __global ushort* dst = (__global ushort*)(sg == 0 ? kc + (size_t)past * D
                                                  : sg == 1 ? vc + (size_t)past * D
                                                            : qb + (sg - 2) * D);
        if (D == 256)
            head_to_f16(hr, hg, cs, sn, dst, 256);
        else
            head_to_f16(hr, hg, cs, sn, dst, 512);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    PROF_DBG(mt, layer, u, 3)

    const float scale = mt->attn_scale;
    const int W = D * 2, HT = jhi + 1;
    float8 acc[8];
    __attribute__((opencl_unroll_hint)) for (int c = 0; c < 8; ++c)
        acc[c] = (float8)(0.0f);
    float8 lsum = (float8)(0.0f);
    float mx[8];
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
        mx[m] = -INFINITY;
    for (int cj = j0 + grp * 16; cj < j1; cj += NGRP * 16) {
        const int j = cj + ln;
        const float madd = j < j1 ? mk[j] : -INFINITY;
        const bool vis = madd > -1e4f;
        if (!sub_group_any(vis))
            continue;
        float8 s = (float8)(0.0f);
        __attribute__((opencl_unroll_hint(XMX_QK_UNROLL))) for (int c = 0; c < D / 16; ++c) {
            ushort8 a;
            uint8 b;
            intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)qb, W, G, W, (int2)(c * 16, 0),
                                                      (__private ushort*)&a);
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)kc, W, HT, W, (int2)(c * 8, cj),
                                                                (__private uint*)&b);
            s = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), s);
        }
        float8 p, cr;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const float sc = fma(s[m], scale, madd);
            const float mn = fmax(mx[m], sub_group_reduce_max(vis ? sc : -INFINITY));
            cr[m] = mx[m] == -INFINITY ? 0.0f : exp(mx[m] - mn);
            p[m] = vis ? exp(sc - mn) : 0.0f;
            mx[m] = mn;
        }
        lsum = fma(lsum, cr, p);
        const short8 pa = as_short8(convert_half8(p));
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < 8; ++c) {
            uint8 b;
            intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void*)vc, W, HT, W,
                                                                 (int2)(dq * 128 + c * 16, cj), (__private uint*)&b);
            acc[c] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(b), acc[c] * cr);
        }
    }
    PROF_DBG(mt, layer, u, 4)

    // Combine the NGRP key groups of each column slice: rescale to the common
    // maximum, then a tree through SLM (acc is [8 heads] per lane = column).
    __local float* slm_acc = slm;                         // [8 slots][8 c][16 lanes][8 heads]
    __local float* msg = slm + 8 * 8 * 128;               // [NSG][8] maxima
    __local float* lsm = msg + NSG * 8;                   // [8 slots][8] sums
    __local float* wt = lsm + 64;                         // attn_merge weights
    if (ln == 0) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
            msg[sg * 8 + m] = mx[m];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float8 w;
    float M[8], l[8];
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
        M[m] = -INFINITY;
        for (int g = 0; g < NGRP; ++g)
            M[m] = fmax(M[m], msg[(g * DQ + dq) * 8 + m]);
        w[m] = mx[m] == -INFINITY ? 0.0f : exp(mx[m] - M[m]);
        l[m] = sub_group_reduce_add(lsum[m]) * w[m];
    }
    __attribute__((opencl_unroll_hint)) for (int c = 0; c < 8; ++c)
        acc[c] *= w;
    for (int half_ = NGRP / 2; half_ >= 1; half_ /= 2) {
        if (grp >= half_ && grp < 2 * half_) {
            const int slot = (grp - half_) * DQ + dq;
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < 8; ++c)
                vstore8(acc[c], (slot * 8 + c) * SG + ln, slm_acc);
            if (ln == 0) {
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
                    lsm[slot * 8 + m] = l[m];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (grp < half_) {
            const int slot = grp * DQ + dq;
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < 8; ++c)
                acc[c] += vload8((slot * 8 + c) * SG + ln, slm_acc);
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
                l[m] += lsm[slot * 8 + m];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const int PS = D + 16;
    if (grp == 0) {
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            if (m >= G)
                break;
            const int qh = kvh * G + m;
            if (NS == 1) {
                // A single split is final already.
                const float il = 1.0f / l[m];
                __global ushort* orow = (__global ushort*)(mt->att + qh * D + dq * 128);
                __attribute__((opencl_unroll_hint)) for (int c = 0; c < 8; ++c)
                    intel_sub_group_block_write_us(orow + c * 16, as_ushort(convert_half(acc[c][m] * il)));
            } else {
                __global float* pp = mt->part + ((size_t)qh * NS + split) * PS;
                __attribute__((opencl_unroll_hint)) for (int c = 0; c < 8; ++c)
                    intel_sub_group_block_write((__global uint*)(pp + dq * 128 + c * 16), as_uint(acc[c][m]));
                if (dq == 0 && ln == 0) {
                    pp[D] = M[m];
                    pp[D + 1] = l[m];
                }
            }
        }
    }
    PROF_DBG(mt, layer, u, 5)
    if (NS > 1)
        attn_merge(mt, layer, kvh, split, NS, G, D, slm, wt, att_base);
    PROF_DBG(mt, layer, u, 6)
}

ATT_FN_ATTR void attn_unit_256(const __global LayerTab* L, const __global ModelTab* mt, const int layer, const int u,
                               const int NS, const int past, const int jlo, const int jhi, const float posf,
                               const __global float* mk, __local float* slm, const int att_base) {
    attn_unit(L, mt, layer, u, NS, past, jlo, jhi, posf, mk, slm, att_base, 256, 256);
}
ATT_FN_ATTR void attn_unit_512(const __global LayerTab* L, const __global ModelTab* mt, const int layer, const int u,
                               const int NS, const int past, const int jlo, const int jhi, const float posf,
                               const __global float* mk, __local float* slm, const int att_base) {
    attn_unit(L, mt, layer, u, NS, past, jlo, jhi, posf, mk, slm, att_base, 512, 512);
}

// Warms L2 with what attn_unit of work-group `u` will read (issued before the
// grid barrier that precedes the attention phase).
inline void attn_prefetch(const __global LayerTab* L, const __global ModelTab* mt, const int u, const int NS,
                          const int jlo, const int jhi) {
    const int HK = L->HK, D = L->D;
    if (u >= HK * NS)
        return;
    const int kvh = u / NS;
    const int2 jr = split_keys(jlo, jhi, NS, u % NS);
    const int cap = mt->kv_cap;
    const __global uint* kc = (const __global uint*)(L->kc + (size_t)kvh * cap * D);
    const __global uint* vc = (const __global uint*)(L->vc + (size_t)kvh * cap * D);
    for (int j = jr.x + get_sub_group_id(); j < jr.y; j += NSG)
        for (int c = 0; c < D / 256; ++c) {
            intel_sub_group_block_prefetch_ui8(kc + (size_t)j * D / 2 + c * 128);
            intel_sub_group_block_prefetch_ui8(vc + (size_t)j * D / 2 + c * 128);
        }
    if (get_sub_group_id() == 0) {
        intel_sub_group_block_prefetch_ui8((const __global uint*)L->gq);
        intel_sub_group_block_prefetch_ui8((const __global uint*)L->gk);
        intel_sub_group_block_prefetch_ui8((const __global uint*)L->invf);
    }
}

// ---------------------------------------------------------------------------
// lm_head: u8 [V_, H] with a per-row scale / zero point, repacked as records of
// 16 rows x 64 bytes (dword j * 16 + lane = row lane, bytes 4j..4j+3). The
// activation is quantised in natural order (dword d of group g = elements
// 64g + 4d .. 4d + 3) as two int8 planes like the int4 GEMVs.
// ---------------------------------------------------------------------------
#define LM_REC_DW 256
#ifndef LM_UNROLL
#define LM_UNROLL 8
#endif

inline void store_qn_l(QB q, float8 v, int b, int K, __local uint* x, int8 hi, int8 lo) {
    x[2 * b] = pack4(hi.s0, hi.s1, hi.s2, hi.s3);
    x[2 * b + 1] = pack4(hi.s4, hi.s5, hi.s6, hi.s7);
    x[K / 4 + 2 * b] = pack4(lo.s0, lo.s1, lo.s2, lo.s3);
    x[K / 4 + 2 * b + 1] = pack4(lo.s4, lo.s5, lo.s6, lo.s7);
    if ((get_sub_group_local_id() & 7) == 0) {
        x[K / 2 + (b >> 3)] = as_uint(q.a);
        x[K / 2 + K / QG + (b >> 3)] = as_uint(q.gsum);
    }
}

// Natural-order variant of quant_slm (same arithmetic as quant_block).
inline void quant_slm_nat(const __local float* src, const int K, __local uint* xq) {
    for (int c0 = get_sub_group_id() * SG; c0 < K / 8; c0 += NSG * SG) {
        const int b = c0 + get_sub_group_local_id();
        const bool ok = b < K / 8;
        const float8 v = ok ? vload8(b, src) : (float8)(0.0f);
        const QB q = quant_block(v, ok);
        const float8 t = v * (1.0f / q.a);
        const float8 hf = rint(t);
        const int8 hi = convert_int8(hf);
        const int8 lo = convert_int8(rint((t - hf) * XL_RES));
        if (ok)
            store_qn_l(q, v, b, K, xq, hi, lo);
    }
}

inline float lm_row_dot(const __global uint* rec, const __local uint* xq) {
    const __local uint* xh = xq;
    const __local uint* xl = xq + H_ / 4;
    const __local float* xa = (const __local float*)(xq + H_ / 2);
    float acc = 0.0f;
    __attribute__((opencl_unroll_hint(LM_UNROLL))) for (int g = 0; g < H_ / QG; ++g) {
        const __global uint* p = rec + g * LM_REC_DW;
        const uint8 w0 = intel_sub_group_block_read8(p);
        const uint8 w1 = intel_sub_group_block_read8(p + 128);
        const uint xhv = intel_sub_group_block_read(xh + g * 16);
        const uint xlv = intel_sub_group_block_read(xl + g * 16);
        int ih = 0, il = 0;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {
            ih += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xhv, j));
            il += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xlv, j));
        }
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {
            ih += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xhv, 8 + j));
            il += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xlv, 8 + j));
        }
        acc = fma(xa[g], fma((float)il, 1.0f / XL_RES, (float)ih), acc);
    }
    return acc;
}

inline float softcap(const __global ModelTab* mt, float y) {
    return mt->cap_div > 0.0f ? tanh(y / mt->cap_div) * mt->cap_mul : y;
}

// ---------------------------------------------------------------------------
// The decode megakernel.
// ---------------------------------------------------------------------------
#define SLM_XF 0                                   // float [H_]
#define SLM_XQ (H_)                                // uint  [XQ_WORDS(8192)]
#define SLM_XQ2 (SLM_XQ + XQ_WORDS(H_))            // second activation (phase D)
#define SLM_XQD (SLM_XQ + XQ_WORDS(TOPK_ * IE_))   // dense down activation (phase E), after the MoE one
#define SLM_JOBS (SLM_XQ + XQ_WORDS(8192))         // Job[24]
#define SLM_TK (SLM_JOBS + 24 * 10)                // int[TOPK_] + float[TOPK_]
#define SLM_TMP (SLM_TK + 2 * TOPK_)               // float[2 * NSG]
#define SLM_XF2 (SLM_TMP + 2 * NSG)                // float [H_], phase D
#define SLM_XF3 (SLM_XF2 + H_)                     // float [H_], phase D
#define SLM_END (SLM_XF3 + H_)
#define SLM_ATT_WORDS (NSG * ATT_QD + 4 * NSG + 512 + ATT_DMAX + 4 + 2 * ATT_DMAX + 16)
#define SLM_WORDS (SLM_END > SLM_ATT_WORDS ? SLM_END : SLM_ATT_WORDS)
#if SLM_XQD + XQ_WORDS(I_) > SLM_JOBS || SLM_XQ2 + XQ_WORDS(H_) > SLM_JOBS
#error activation buffers overlap
#endif

REQD __kernel void mk_decode(const __global ModelTab* mt, const __global LayerTab* lt,
                             const __global float* embeds, const __global int* pos, const __global int* am,
                             __global float* logits, const int past, const int L, const int epoch, const int ns_s,
                             const int ns_f, const int att_base_s, const int att_base_f, const int xmx) {
    __local uint slm[SLM_WORDS];
    __local float* xf = (__local float*)(slm + SLM_XF);
    __local uint* xq = slm + SLM_XQ;
    __local Job* jobs = (__local Job*)(slm + SLM_JOBS);
    __local int* tk_i = (__local int*)(slm + SLM_TK);
    __local float* tk_w = (__local float*)(slm + SLM_TK + TOPK_);
    __local float* tmp = (__local float*)(slm + SLM_TMP);

    const int wg = get_group_id(0);
    const int sg = get_sub_group_id();
    const int ln = get_sub_group_local_id();
    const int lid = get_local_id(0);
    const int gs = wg * NSG + sg;
    const int target = epoch * NWG;
    __global int* sync = mt->sync;
    __global int* err = mt->err;
    int bar = 0;
#define GRID_SYNC()                              \
    {                                            \
        grid_arrive(sync + bar);                 \
        PROF_MARK(mt, bar, 0)                    \
        grid_wait(sync + bar, target, err);      \
        PROF_MARK(mt, bar, 1)                    \
        ++bar;                                   \
    }
    PROF_MARK(mt, NL_ * NPH + 1, 0)

    const float pos0 = (float)pos[0];
    // Key window of the current query (mask MegaKernel semantics, query = past).
    const int jhi = min(L - 1, past + (int)mt->causal_off - 1);
    const int jlo_s = max(0, past - (int)mt->window + 1);
    float ls_prev = 0.0f;  // layer scalar of layer l - 1, loaded during its phase E

    // Additive key mask, built once from attention_mask (a graph input that may
    // live in host memory): read per key in every attention unit, its load
    // latency serialised the key loop. Ordered before phase B by A's barrier.
    {
        const int ams = mt->am_stride;
        for (int j = wg * LWS + lid; j < L; j += NWG * LWS)
            mt->kmask[j] = am ? (float)convert_half_rte(fma((float)am[(size_t)j * ams], mt->p_mul, mt->p_add)) : 0.0f;
    }

    for (int l = 0; l < NL_; ++l) {
        const __global LayerTab* Ly = lt + l;
        const int R = Ly->R;
        __global int* pool = sync + NL_ * NPH + 1 + NL_ + l * 4;  // run_jobs_pool counters of A, C, D, E

        // ---------------- A: residual, input RMS, q|k|v ----------------
        {
            float xv[HPT_], gv[HPT_];
            if (l == 0) {
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                    xv[i] = (float)convert_half(embeds[lid + i * LWS]);
                    gv[i] = vload_half(lid + i * LWS, Ly->w_in);
                }
            } else {
                ffn_out_vals(lt + l - 1, mt, Ly->w_in, ls_prev, xv, gv, tmp);
            }
            if (wg == 0) {
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                    vstore_half(xv[i], lid + i * LWS, mt->xbuf);
#ifdef MK_DEBUG
                    vstore_half(xv[i], lid + i * LWS, (__global half*)mt->prof + (size_t)l * H_);
#endif
                }
            }
            rms_to_slm_v(xv, gv, xf, tmp);
            barrier(CLK_LOCAL_MEM_FENCE);
            quant_slm(xf, H_, xq);
            if (lid == 0) {
                Job j = {Ly->qkv, mt->raw, H_ / QG, R / SG * (H_ / QG), SLM_XQ, H_, 0, 1.0f};
                jobs[0] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            run_jobs_pool(jobs, 1, gs, NSGT, slm, POOL_PCT, pool + 0, epoch);
#ifndef NO_ATT_PF
            attn_prefetch(Ly, mt, wg, Ly->sliding ? ns_s : ns_f, Ly->sliding ? jlo_s : 0, jhi);
#endif
        }
        GRID_SYNC();

        // ---------------- B: attention ----------------
        // Zero accumulators whose last reader has passed: gu, egu (layer l-1),
        // m, e (read in A).
        zero_grid(mt->gu, 2 * I_);
        zero_grid(mt->egu, TOPK_ * 2 * IE_);
        zero_grid(mt->m, H_);
        zero_grid(mt->e, H_);
        {
            const int NS = Ly->sliding ? ns_s : ns_f;
            const int jlo = Ly->sliding ? jlo_s : 0;
            const float posf = Ly->pos_f16 ? (float)convert_half_rte(pos0) : pos0;
            if (wg < Ly->HK * NS) {
                // Long key ranges go to the XMX unit; on a few keys per unit
                // its fixed costs (q round trip through memory, deeper
                // combine) lose to the scalar one.
                if ((xmx >> (Ly->sliding ? 0 : 1)) & 1)
                    attn_unit_xmx(Ly, mt, l, wg, NS, past, jlo, jhi, posf, mt->kmask, xf,
                                  Ly->sliding ? att_base_s : att_base_f, Ly->D);
                else if (Ly->D == 256)
                    attn_unit_256(Ly, mt, l, wg, NS, past, jlo, jhi, posf, mt->kmask, xf, att_base_s);
                else
                    attn_unit_512(Ly, mt, l, wg, NS, past, jlo, jhi, posf, mt->kmask, xf, att_base_f);
            }
            else {
                // DRAM is idle during attention: the other work-groups pull
                // (up to OPROJ_PF_MAX records of) o_proj into L2.
                const int busy = Ly->HK * NS * NSG;
                const int tot = min(OPROJ_PF_MAX, H_ / SG * (Ly->HQ * Ly->D / QG));
                const int r0 = slice_lo(tot, gs - busy, NSGT - busy), r1 = slice_lo(tot, gs - busy + 1, NSGT - busy);
                prefetch_recs(Ly->oproj + (size_t)r0 * REC_DW, r1 - r0);
            }
        }
        GRID_SYNC();

        // ---------------- C: o_proj ----------------
        zero_grid(mt->raw, R);
        {
            const int KA = Ly->HQ * Ly->D;
            // Gammas and router matrix of D's prologue.
            if (gs >= NSGT - 4) {
                const int k = NSGT - 1 - gs;
                prefetch_bytes(k == 0 ? Ly->w_pa : k == 1 ? Ly->w_pf2 : k == 2 ? Ly->w_pf : Ly->w_r, H_ * 2);
            }
            {
                const int n = E_ * H_ * 2 / 512;
                for (int i = gs; i < n; i += NSGT)
                    prefetch_bytes((const __global uchar*)Ly->rw + (size_t)i * 512, 512);
            }
#ifdef MK_DEBUG
            if (wg == 0)
                for (int i = lid; i < KA; i += LWS)
                    ((__global half*)mt->prof)[(size_t)NL_ * H_ + (size_t)l * 8192 + i] = mt->att[i];
#endif
            const __global half* att = mt->att;
#define LOAD_ATT(b) vload_half8((b), att)
            QUANT_GLOBAL_WITH(LOAD_ATT, KA, xq)
#undef LOAD_ATT
            if (lid == 0) {
                Job j = {Ly->oproj, mt->o, KA / QG, H_ / SG * (KA / QG), SLM_XQ, KA, 0, 1.0f};
                jobs[0] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            run_jobs_pool(jobs, 1, gs, NSGT, slm, POOL_PCT, pool + 1, epoch);
            // Next: the dense gate/up projections of D (same slices as its run_jobs).
            {
                const int per = I_ / SG * (H_ / QG);
                const int r0 = slice_lo(2 * per, gs, NSGT);
                const int r1 = slice_lo(2 * per, gs + 1, NSGT);
                const __global uint* m0 = r0 < per ? Ly->gate + (size_t)r0 * REC_DW : Ly->up + (size_t)(r0 - per) * REC_DW;
                prefetch_recs(m0, min(PF_RECS, min(r1, r0 < per ? per : 2 * per) - r0));
            }
        }
        GRID_SYNC();

        // ---------------- D: h1, router, dense gate/up, MoE gate/up ----------------
        // Every work-group first adds its slice of the router logits, then
        // streams its dense gate/up share; by then all router slices are in,
        // so the MoE gate/up projections follow inside the same phase.
        {
            const __global long* o = mt->o;
            const __global half* xb = mt->xbuf;
            const __global half* w_pa = Ly->w_pa;
            const __global half* w_pf2 = Ly->w_pf2;
            const __global half* w_pf = Ly->w_pf;
            const __global half* w_r = Ly->w_r;
            __local float* xf2 = (__local float*)(slm + SLM_XF2);
            __local float* xf3 = (__local float*)(slm + SLM_XF3);
            float hv[HPT_], xbv[HPT_], ga[HPT_], g2[HPT_], gf[HPT_], gr[HPT_];
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                const int k = lid + i * LWS;
                hv[i] = fx2f(o[k]);
                xbv[i] = vload_half(k, xb);
                ga[i] = vload_half(k, w_pa);
                g2[i] = vload_half(k, w_pf2);
                gf[i] = vload_half(k, w_pf);
                gr[i] = vload_half(k, w_r);
            }
            float ss = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
                ss += hv[i] * hv[i];
            const float inv_o = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
            ss = 0.0f;
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                hv[i] = xbv[i] + hv[i] * inv_o * ga[i];
                ss += hv[i] * hv[i];
            }
            if (wg == 0) {
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
                    mt->h1[lid + i * LWS] = hv[i];
            }
            const float inv1 = rsqrt(wg_sum(ss, tmp) / H_ + EPS_);
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i) {
                xf[lid + i * LWS] = hv[i] * inv1 * g2[i];
                xf2[lid + i * LWS] = hv[i] * inv1 * gf[i];
                xf3[lid + i * LWS] = hv[i] * inv1 * gr[i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            quant_slm(xf, H_, slm + SLM_XQ2);
            quant_slm(xf2, H_, xq);
            if (lid < 2) {
                Job j = {lid ? Ly->up : Ly->gate, mt->gu + lid * I_, H_ / QG, I_ / SG * (H_ / QG), SLM_XQ, H_, 0,
                         1.0f};
                jobs[lid] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            // Router: f16 [E_, H_], items of 16 rows x 16 columns (row per lane).
            {
                const int NIT = (E_ / SG) * (H_ / SG);
                int i = slice_lo(NIT, gs, NSGT);
                const int i1 = slice_lo(NIT, gs + 1, NSGT);
                while (i < i1) {
                    const int rb = i / (H_ / SG);
                    const int e = min(i1, (rb + 1) * (H_ / SG));
                    const __global half* row = Ly->rw + (size_t)(rb * SG + ln) * H_;
                    float acc = 0.0f;
                    for (; i < e; ++i) {
                        const int kc = i - rb * (H_ / SG);
                        const float16 w = vload_half16(kc, row);
                        const float x = xf3[kc * SG + ln];
                        __attribute__((opencl_unroll_hint)) for (int j = 0; j < SG; ++j)
                            acc = fma(w[j], sub_group_broadcast(x, j), acc);
                    }
                    atomic_addfx(mt->lg + rb * SG + ln, acc);
                }
            }
            // Routing overlaps the dense gate/up: sub-group 0 of every
            // work-group waits for all router slices, runs the top-k and posts
            // the MoE jobs while the others stream, so it takes a smaller
            // dense share (ROUTE_SHARE / 16 of a regular one).
            __global int* rdone = sync + NL_ * NPH + 1 + l;
            grid_arrive(rdone);
            __local Job* mjobs = jobs + 2;
            {
                const int nd = 2 * (I_ / SG * (H_ / QG));
                const int w0 = slice_lo(nd, wg, NWG), nw = slice_lo(nd, wg + 1, NWG) - w0;
                const int n0 = nw * ROUTE_SHARE / (SG * (NSG - 1) + ROUTE_SHARE);
                const int r0 = sg == 0 ? w0 : w0 + n0 + slice_lo(nw - n0, sg - 1, NSG - 1);
                const int r1 = sg == 0 ? w0 + n0 : w0 + n0 + slice_lo(nw - n0, sg, NSG - 1);
                if (sg == 0) {
                    sg_wait(rdone, target, err);
                    topk_local(mt->lg, tk_i, tk_w);
                    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
                    const int t = ln % TOPK_, up = ln / TOPK_;
                    const size_t eoff = (size_t)tk_i[t] * (IE_ / SG) * (H_ / QG) * REC_DW;
                    Job j = {(up ? Ly->eup : Ly->egate) + eoff, mt->egu + (t * 2 + up) * IE_, H_ / QG,
                             IE_ / SG * (H_ / QG), SLM_XQ2, H_, 0, 1.0f};
                    mjobs[up * TOPK_ + t] = j;
                }
                run_jobs_range(jobs, 2, r0, r1, slm);
            }
            PROF_MARK(mt, bar, 3)
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 4)
            run_jobs_pool(mjobs, 2 * TOPK_, gs, NSGT, slm, POOL_PCT, pool + 2, epoch);
            // Next: E's down projections (experts known now).
            {
                const int per = H_ / SG * (IE_ / QG);
                const int tot = TOPK_ * per + H_ / SG * (I_ / QG);
                const int r0 = slice_lo(tot, gs, NSGT), r1 = slice_lo(tot, gs + 1, NSGT);
                if (r0 < TOPK_ * per) {
                    const int t = r0 / per;
                    prefetch_recs(Ly->edown + ((size_t)tk_i[t] * per + (r0 - t * per)) * REC_DW,
                                  min(PF_RECS, min(r1, (t + 1) * per) - r0));
                } else {
                    prefetch_recs(Ly->down + (size_t)(r0 - TOPK_ * per) * REC_DW, min(PF_RECS, r1 - r0));
                }
            }
        }
        GRID_SYNC();

        // ---------------- E: MoE down, dense down ----------------
        zero_grid(mt->o, H_);
        zero_grid(mt->lg, E_);
        {
            ls_prev = vload_half(0, Ly->ls);
            // Gammas of the next prologue (ffn_out_vals + RMS).
            if (gs >= NSGT - 4) {
                const int k = NSGT - 1 - gs;
                prefetch_bytes(k == 0   ? Ly->w_pf1
                               : k == 1 ? Ly->w_pf2p
                               : k == 2 ? Ly->w_post
                               : l + 1 < NL_ ? lt[l + 1].w_in : mt->w_final,
                               H_ * 2);
            }
            // gelu(gate) * up, quantised into SLM: the MoE slots (TOPK_ * IE_)
            // at SLM_XQ, the dense vector (I_) at SLM_XQD. Group gi < NGM is
            // MoE group gi, the rest dense. A work-group only builds the
            // contiguous group range [ga, gb) of its static record slice,
            // plus the dense vector, which holds run_jobs_pool's shared tail.
            {
                const int NGM = TOPK_ * IE_ / QG, NGD = I_ / QG;
                const int per = H_ / SG * (IE_ / QG);
                const int total = TOPK_ * per + H_ / SG * (I_ / QG);
                const int S = pool_static(total, POOL_PCT_E);
                const int r0 = slice_lo(S, wg * NSG, NSGT), r1 = slice_lo(S, wg * NSG + NSG, NSGT);
                const int ga = r0 < TOPK_ * per ? r0 / per * (IE_ / QG) : NGM;
                const int gb = r1 > TOPK_ * per ? NGM + NGD : ((r1 - 1) / per + 1) * (IE_ / QG);
                const int n1 = gb - ga, n = n1 + (gb < NGM + NGD && S < total ? NGD : 0);
                for (int g2 = sg * 2; g2 < n; g2 += NSG * 2) {
                    const int idx = g2 + ln / 8;
                    const bool ok = idx < n;
                    const int gi = idx < n1 ? ga + idx : NGM + idx - n1;
                    const bool moe = gi < NGM;
                    const int gl = moe ? gi : gi - NGM;         // group within its vector
                    const int b = gl * 8 + (ln & 7);            // 8-block within its vector
                    float8 v = (float8)(0.0f);
                    if (ok) {
                        const __global long* gp = moe ? mt->egu + (b / (IE_ / 8)) * 2 * IE_ : mt->gu;
                        const int c = moe ? b % (IE_ / 8) : b;
                        v = gelu_tanh8(fx2f8(vload8(c, gp))) * fx2f8(vload8(c, gp + (moe ? IE_ : I_)));
                    }
                    const QB q = quant_block(v, ok);
                    if (ok)
                        store_qb_l(q, b, moe ? TOPK_ * IE_ : I_, moe ? xq : slm + SLM_XQD);
                }
            }
            if (lid < TOPK_) {
                const size_t eoff = (size_t)tk_i[lid] * (H_ / SG) * (IE_ / QG) * REC_DW;
                Job j = {Ly->edown + eoff, mt->e, IE_ / QG, H_ / SG * (IE_ / QG), SLM_XQ, TOPK_ * IE_,
                         lid * (IE_ / QG), tk_w[lid]};
                jobs[lid] = j;
            } else if (lid == TOPK_) {
                Job j = {Ly->down, mt->m, I_ / QG, H_ / SG * (I_ / QG), SLM_XQD, I_, 0, 1.0f};
                jobs[TOPK_] = j;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            PROF_MARK(mt, bar, 2)
            run_jobs_pool(jobs, TOPK_ + 1, gs, NSGT, slm, POOL_PCT_E, pool + 3, epoch);
            // Next: q|k|v of the next layer.
            if (l + 1 < NL_)
                prefetch_single(lt[l + 1].qkv, lt[l + 1].R / SG * (H_ / QG), gs, NSGT);
        }
        GRID_SYNC();
    }

    // ---------------- G: final norm, lm_head, soft-cap ----------------
    zero_grid(mt->egu, TOPK_ * 2 * IE_);
    {
        float xv[HPT_], gv[HPT_];
        ffn_out_vals(lt + NL_ - 1, mt, mt->w_final, ls_prev, xv, gv, tmp);
        rms_to_slm_v(xv, gv, xf, tmp);
        barrier(CLK_LOCAL_MEM_FENCE);
        quant_slm_nat(xf, H_, xq);
        barrier(CLK_LOCAL_MEM_FENCE);
        float sx = 0.0f;
        const __local float* xs = (const __local float*)(xq + H_ / 2 + H_ / QG);
        for (int g = 0; g < H_ / QG; ++g)
            sx += xs[g];
        PROF_MARK(mt, bar, 2)
        const int nrb = V_ / SG;
        const int rb0 = slice_lo(nrb, gs, NSGT), rb1 = slice_lo(nrb, gs + 1, NSGT);
        for (int rb = rb0; rb < rb1; ++rb) {
            const float acc = lm_row_dot(mt->lm + (size_t)rb * (H_ / QG) * LM_REC_DW, xq);
            const int row = rb * SG + ln;
            const float y = (float)mt->lm_s[row] * (acc - (float)mt->lm_z[row] * sx);
            logits[row] = softcap(mt, y);
        }
    }
    PROF_MARK(mt, NL_ * NPH + 1, 1)
#undef GRID_SYNC
}

// ===========================================================================
// Prompt (multi-token) helpers of the whole-model path.
// ===========================================================================
__kernel void mk_cvt_embeds(const __global float* in, __global half* out, const int n) {
    const int i = get_global_id(0);
    if (i < n)
        out[i] = convert_half(in[i]);
}

// K / V heads of attn_post's head-major output [HQ + 2 HK][T][D] -> cache rows past..past+T-1.
__kernel void mk_kv_append(const __global half* hm, const int T, const int HQ, const int HK, const int D,
                           __global half* kc, __global half* vc, const int cap, const int past) {
    const int d = get_global_id(0);
    const int t = get_global_id(1);
    const int h = get_global_id(2);
    if (d >= D || t >= T)
        return;
    const int kv = h % HK;
    const __global half* src = hm + ((size_t)(HQ + h) * T + t) * D;
    __global half* dst = (h < HK ? kc : vc) + ((size_t)kv * cap + past + t) * D;
    dst[d] = src[d];
}

// First and last visible key of every query row of a [T, L] f16 mask.
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_mask_range(const __global half* mask, const int T, const int L, __global int* lo, __global int* hi) {
    const int t = get_global_id(1);
    if (t >= T)
        return;
    const __global half* row = mask + (size_t)t * L;
    int a = L, b = -1;
    for (int j = get_sub_group_local_id(); j < L; j += SG) {
        if ((float)row[j] > -1e4f) {
            a = min(a, j);
            b = max(b, j);
        }
    }
    a = sub_group_reduce_min(a);
    b = sub_group_reduce_max(b);
    if (get_sub_group_local_id() == 0) {
        lo[t] = a;
        hi[t] = b;
    }
}

// Prompt attention: one sub-group per (head, QB = 512 / D consecutive queries),
// online softmax over the visible keys of the cache with the explicit mask.
// q comes from attn_post's head-major output, O is [T][HQ * D]. D is a literal
// at both call sites.
inline void pf_attn_body(const __global half* hm, const __global half* kcache, const __global half* vcache,
                         const __global half* mask, const __global int* mlo, const __global int* mhi, const int T,
                         const int L, const int HQ, const int HK, const int cap, const float scale,
                         __global half* O, const int D) {
    const int QB = ATT_QD / D;
    const int nj = D / SG;
    const int nqb = (T + QB - 1) / QB;
    const int w = get_group_id(0) * (get_local_size(0) / SG) + get_sub_group_id();
    if (w >= HQ * nqb)
        return;
    const int h = w / nqb, t0 = (w % nqb) * QB;
    const int kvh = h / (HQ / HK);
    const int ln = get_sub_group_local_id();
    float qv[ATT_QD / SG], acc[ATT_QD / SG];
    float mx[2], ls[2];
    int jl = L, jh = -1;
    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
        const int t = min(t0 + q, T - 1);
        jl = min(jl, mlo[t]);
        jh = max(jh, mhi[t]);
        mx[q] = -INFINITY;
        ls[q] = 0.0f;
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j) {
            qv[q * nj + j] = (float)hm[((size_t)h * T + t) * D + j * SG + ln] * scale;
            acc[q * nj + j] = 0.0f;
        }
    }
    const __global half* kc = kcache + (size_t)kvh * cap * D;
    const __global half* vc = vcache + (size_t)kvh * cap * D;
    // Keys go in chunks of SG: every lane forms the per-lane partial dots of
    // all SG keys, a butterfly transpose-reduce leaves key `ln`'s score on lane
    // ln (15 shuffles instead of SG full reductions), and the online softmax
    // is updated once per chunk instead of once per key.
    for (int j0 = jl; j0 <= jh; j0 += SG) {
        const int jk = j0 + ln;
        float madd[2];
        bool any = false;
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
            const int t = t0 + q;
            madd[q] = (t < T && jk <= jh) ? (float)mask[(size_t)t * L + jk] : -INFINITY;
            any |= madd[q] > -1e4f;
        }
        if (!sub_group_any(any))
            continue;
        float v[2][SG];
        __attribute__((opencl_unroll_hint)) for (int k = 0; k < SG; ++k) {
            const __global ushort* kp = (const __global ushort*)(kc + (size_t)min(j0 + k, jh) * D);
            float s[2] = {0.0f, 0.0f};
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                const half8 k8 = as_half8(intel_sub_group_block_read_us8(kp + c * 128));
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q)
                        s[q] = fma(qv[q * nj + c * 8 + i], (float)k8[i], s[q]);
            }
            __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q)
                v[q][k] = s[q];
        }
        float sc[2];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
            __attribute__((opencl_unroll_hint)) for (int hb = SG / 2; hb >= 1; hb >>= 1) {
                const bool up = (ln & hb) != 0;
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < hb; ++i) {
                    const float send = up ? v[q][i] : v[q][i + hb];
                    const float keep = up ? v[q][i + hb] : v[q][i];
                    v[q][i] = keep + intel_sub_group_shuffle_xor(send, (uint)hb);
                }
            }
            sc[q] = v[q][0] + madd[q];
        }
        float pr[2];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
            const bool vis = madd[q] > -1e4f;
            const float mn = fmax(mx[q], sub_group_reduce_max(vis ? sc[q] : -INFINITY));
            const float cr = mx[q] == -INFINITY ? 0.0f : exp(mx[q] - mn);
            pr[q] = vis ? exp(sc[q] - mn) : 0.0f;
            ls[q] = fma(ls[q], cr, sub_group_reduce_add(pr[q]));
            mx[q] = mn;
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
                acc[q * nj + j] *= cr;
        }
        const int nk = min(SG, jh - j0 + 1);
        for (int k = 0; k < nk; ++k) {
            float pk[2];
            bool nz = false;
            __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
                pk[q] = sub_group_broadcast(pr[q], k);
                nz |= pk[q] != 0.0f;
            }
            if (!nz)
                continue;
            const __global ushort* vp = (const __global ushort*)(vc + (size_t)(j0 + k) * D);
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < nj / 8; ++c) {
                const half8 v8 = as_half8(intel_sub_group_block_read_us8(vp + c * 128));
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 8; ++i)
                    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
                        const int kk = q * nj + c * 8 + i;
                        acc[kk] = fma(pk[q], (float)v8[i], acc[kk]);
                    }
            }
        }
    }
    __attribute__((opencl_unroll_hint)) for (int q = 0; q < QB; ++q) {
        const int t = t0 + q;
        if (t < T) {
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < nj; ++j)
                O[(size_t)t * HQ * D + h * D + j * SG + ln] = convert_half(acc[q * nj + j] / ls[q]);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void mk_pf_attn(const __global half* hm, const __global half* kcache, const __global half* vcache,
                         const __global half* mask, const __global int* mlo, const __global int* mhi, const int T,
                         const int L, const int HQ, const int HK, const int D, const int cap, const float scale,
                         __global half* O) {
    if (D == 256)
        pf_attn_body(hm, kcache, vcache, mask, mlo, mhi, T, L, HQ, HK, cap, scale, O, 256);
    else
        pf_attn_body(hm, kcache, vcache, mask, mlo, mhi, T, L, HQ, HK, cap, scale, O, 512);
}

// XMX flash attention for the prompt path. One sub-group = one head, 8
// consecutive queries and a 256-wide slice of the output (D = 512 heads use
// two sub-groups that both compute the scores). Per 16-key chunk:
//   S[8x16] = Q[8xD] K^T   DPAS; Q tiles are 2D block reads of hm (rows past T
//                          read 0), K^T comes from a transposed 32-bit read
//   online softmax per query row (one max reduction per row and chunk)
//   O[8x256] += P V        DPAS; V tiles via the VNNI-transforming read
// Rows past jh are outside the read height and come back as zeros.
#define FA_SGS 8
__attribute__((intel_reqd_sub_group_size(SG))) __attribute__((reqd_work_group_size(SG * FA_SGS, 1, 1)))
__kernel void mk_pf_fa(const __global half* hm, const __global half* kcache, const __global half* vcache,
                       const __global half* mask, const __global int* mlo, const __global int* mhi, const int T,
                       const int L, const int HQ, const int HK, const int D, const int cap, const float scale,
                       __global half* O) {
    const int ndh = D / 256;
    const int nqb = (T + 7) / 8;
    const int w = get_group_id(0) * FA_SGS + get_sub_group_id();
    if (w >= HQ * nqb * ndh)
        return;
    const int dh = w % ndh, qb = (w / ndh) % nqb, h = w / (ndh * nqb);
    const int t0 = qb * 8;
    const int kvh = h / (HQ / HK);
    const int ln = get_sub_group_local_id();
    int jl = L, jh = -1;
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
        const int t = min(t0 + m, T - 1);
        jl = min(jl, mlo[t]);
        jh = max(jh, mhi[t]);
    }
    const __global half* qp = hm + (size_t)h * T * D;
    const __global half* kc = kcache + (size_t)kvh * cap * D;
    const __global half* vc = vcache + (size_t)kvh * cap * D;
    const int W = D * 2, HT = jh + 1;
    float8 acc[16];
    __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c)
        acc[c] = (float8)(0.0f);
    float8 lsum = (float8)(0.0f);
    float mx[8];
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m)
        mx[m] = -INFINITY;
    for (int j0 = jl; j0 <= jh; j0 += 16) {
        const int j = j0 + ln;
        float8 madd;
        int vany = 0;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const int t = t0 + m;
            madd[m] = (t < T && j <= jh) ? (float)mask[(size_t)t * L + j] : -INFINITY;
            vany |= madd[m] > -1e4f;
        }
        if (!sub_group_any(vany))
            continue;
        float8 s = (float8)(0.0f);
        for (int c = 0; c < D / 16; ++c) {
            ushort8 a;
            uint8 b;
            intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)qp, W, T, W, (int2)(c * 16, t0),
                                                      (__private ushort*)&a);
            intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)kc, W, HT, W, (int2)(c * 8, j0),
                                                                (__private uint*)&b);
            s = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), as_int8(b), s);
        }
        float8 p, cr;
        int rescale = 0;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
            const bool vis = madd[m] > -1e4f;
            const float sc = fma(s[m], scale, madd[m]);
            const float mn = fmax(mx[m], sub_group_reduce_max(vis ? sc : -INFINITY));
            cr[m] = mx[m] == -INFINITY ? 0.0f : exp(mx[m] - mn);
            rescale |= cr[m] != 1.0f;
            p[m] = vis ? exp(sc - mn) : 0.0f;
            mx[m] = mn;
        }
        lsum = fma(lsum, cr, p);
        if (rescale)
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c)
                acc[c] *= cr;
        const short8 pa = as_short8(convert_half8(p));
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c) {
            uint8 b;
            intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void*)vc, W, HT, W,
                                                                 (int2)(dh * 256 + c * 16, j0), (__private uint*)&b);
            acc[c] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(b), acc[c]);
        }
    }
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; ++m) {
        const int t = t0 + m;
        if (t >= T)
            break;
        const float l = sub_group_reduce_add(lsum[m]);
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        __global ushort* orow = (__global ushort*)(O + (size_t)t * HQ * D + h * D + dh * 256);
        __attribute__((opencl_unroll_hint)) for (int c = 0; c < 16; ++c)
            intel_sub_group_block_write_us(orow + c * 16, as_ushort(convert_half(acc[c][m] * inv)));
    }
}

// Final norm + lm_head + soft-cap for `rows` tokens of a [rows, H] f16 residual.
// Work-group: LM_TB tokens x 16 sub-groups, each sub-group LM_RB row blocks.
#define LM_TB 4
#define LM_RB 4
REQD __kernel void mk_lm_head_rows(const __global ModelTab* mt, const __global half* x, const int rows,
                                   __global float* logits) {
    __local uint xq[LM_TB][XQ_WORDS(H_)];
    __local float xf[H_];
    __local float tmp[NSG];
    __local float sxs[LM_TB];
    // Token groups vary fastest across work-groups, so the groups in flight
    // share one weight slice in L2 instead of each streaming all of lm_head.
    const int t0 = get_group_id(0) * LM_TB;
    const int lid = get_local_id(0);
    for (int q = 0; q < LM_TB; ++q) {
        const int t = min(t0 + q, rows - 1);
        float v[HPT_];
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < HPT_; ++i)
            v[i] = vload_half(lid + i * LWS, x + (size_t)t * H_);
        rms_to_slm(v, mt->w_final, xf, tmp);
        barrier(CLK_LOCAL_MEM_FENCE);
        quant_slm_nat(xf, H_, xq[q]);
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            float sx = 0.0f;
            const __local float* xs = (const __local float*)(xq[q] + H_ / 2 + H_ / QG);
            for (int g = 0; g < H_ / QG; ++g)
                sx += xs[g];
            sxs[q] = sx;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int ln = get_sub_group_local_id();
    const int rb0 = (get_group_id(1) * NSG + get_sub_group_id()) * LM_RB;
    for (int rb = rb0; rb < min(rb0 + LM_RB, V_ / SG); ++rb) {
        const __global uint* rec = mt->lm + (size_t)rb * (H_ / QG) * LM_REC_DW;
        float acc[LM_TB];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < LM_TB; ++q)
            acc[q] = 0.0f;
        for (int g = 0; g < H_ / QG; ++g) {
            const __global uint* p = rec + g * LM_REC_DW;
            const uint8 w0 = intel_sub_group_block_read8(p);
            const uint8 w1 = intel_sub_group_block_read8(p + 128);
            __attribute__((opencl_unroll_hint)) for (int q = 0; q < LM_TB; ++q) {
                const uint xhv = intel_sub_group_block_read(xq[q] + g * 16);
                const uint xlv = intel_sub_group_block_read(xq[q] + H_ / 4 + g * 16);
                int ih = 0, il = 0;
                __attribute__((opencl_unroll_hint)) for (int j = 0; j < 8; ++j) {
                    ih += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xhv, j));
                    il += dot_4x8packed_us_int(w0[j], sub_group_broadcast(xlv, j));
                    ih += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xhv, 8 + j));
                    il += dot_4x8packed_us_int(w1[j], sub_group_broadcast(xlv, 8 + j));
                }
                const float a = ((const __local float*)(xq[q] + H_ / 2))[g];
                acc[q] = fma(a, fma((float)il, 1.0f / XL_RES, (float)ih), acc[q]);
            }
        }
        const int row = rb * SG + ln;
        const float s = (float)mt->lm_s[row], z = (float)mt->lm_z[row];
        __attribute__((opencl_unroll_hint)) for (int q = 0; q < LM_TB; ++q) {
            const int t = t0 + q;
            if (t < rows)
                logits[(size_t)t * V_ + row] = softcap(mt, s * (acc[q] - z * sxs[q]));
        }
    }
}
)CLC";

// Host-side launch geometry of mk_lm_head_rows (must match LM_TB / LM_RB).
constexpr int kLmTb = 4;
constexpr int kLmRb = 4;

namespace mk {
namespace {

#define CL_OK(expr, what)                                                  \
    do {                                                                   \
        cl_int _e = (expr);                                                \
        if (_e != CL_SUCCESS) {                                            \
            std::fprintf(stderr, "[Gemma4] %s failed: %d\n", what, _e);    \
            return _e;                                                     \
        }                                                                  \
    } while (0)

cl_program build_sources(cl_context ctx, cl_device_id dev, std::initializer_list<const char*> srcs, const char* opts,
                         cl_int* err) {
    std::vector<const char*> v(srcs);
    cl_program prog = clCreateProgramWithSource(ctx, static_cast<cl_uint>(v.size()), v.data(), nullptr, err);
    if (*err != CL_SUCCESS)
        return nullptr;
    *err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    size_t n = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::vector<char> log(n + 1, 0);
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    if (*err != CL_SUCCESS || std::getenv("OV_MEGAKERNEL_DUMP"))
        std::fprintf(stderr, "[Gemma4] build %s:\n%s\n", *err == CL_SUCCESS ? "log" : "FAILED", log.data());
    if (*err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return nullptr;
    }
    return prog;
}

cl_program build(cl_context ctx, cl_device_id dev, const char* src, const char* opts, cl_int* err) {
    return build_sources(ctx, dev, {src}, opts, err);
}

// All 30 kFfn instances share one program per context; pf_mb > 0 selects the
// prefill program built with that GEMM tile height.
cl_program ffn_program(cl_context ctx, cl_device_id dev, cl_int* err, int pf_mb = 0) {
    static std::mutex mu;
    static std::map<std::pair<cl_context, int>, cl_program> cache;
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find({ctx, pf_mb});
    if (it != cache.end()) {
        *err = CL_SUCCESS;
        return it->second;
    }
    char opts[512];
    // OV_MEGAKERNEL_CL_OPTS: extra -D overrides for tuning experiments.
    const char* extra = std::getenv("OV_MEGAKERNEL_CL_OPTS");
    std::snprintf(opts, sizeof(opts), "-DH_=%d -DI_=%d -DIE_=%d -DE_=%d -DTOPK_=%d -DEPS_=1e-6f %s %s",
                  kHidden, kInter, kExpertInter, kExperts, kTopK, kFfnBuildOpts, extra ? extra : "");
    std::string o = opts;
    if (pf_mb)
        o += " -DPF_MB=" + std::to_string(pf_mb) + " -DPF_NSG=" + std::to_string(kPfNsg) + " " + kPrefillBuildOpts;
    cl_program p = pf_mb ? build_sources(ctx, dev, {kFfnSource, kPrefillSource}, o.c_str(), err)
                         : build(ctx, dev, kFfnSource, o.c_str(), err);
    if (p)
        cache.emplace(std::make_pair(ctx, pf_mb), p);
    return p;
}

// Multi-token intermediates, shared by every MegaKernel of a context: the
// layers run one after another on the same in-order queue.
enum PfBuf {
    kPfAT,    // tiled f16 GEMM input (q|k|v or o_proj)
    kPfRaw,   // f32 [T, R] q|k|v projections
    kPfO,     // f32 [T, H] o_proj
    kPfH1,    // f32 [T, H]
    kPfXD,    // tiled f16 [T, H] dense MLP input
    kPfXR,    // tiled f16 [T, H] router input
    kPfX2,    // f16 [T, H] MoE input
    kPfLog,   // f32 [KS, T, E] router partials
    kPfTkI,   // i32 [T, TOPK]
    kPfTkW,   // f32 [T, TOPK]
    kPfAct,   // tiled f16 [T, I]
    kPfMD,    // f32 [T, H] dense MLP output
    kPfHdr,   // i32 [1] MoE tile count
    kPfTExp,  // i32 [tiles]
    kPfPair,  // i32 [tiles * MT]
    kPfXE,    // tiled f16 [tiles * MT, H]
    kPfAE,    // tiled f16 [tiles * MT, IE]
    kPfYS,    // f16 [T * TOPK, H] weighted expert outputs
    kPfBufs
};
struct PfScratch {
    void* buf[kPfBufs] = {};
    size_t cap[kPfBufs] = {};
    int refs = 0;
};
std::mutex g_pf_mu;
std::map<cl_context, PfScratch> g_pf;

enum PfKernel { kPfAttnRms, kPfGemm, kPfTile, kPfFfnIn, kPfGemmF16w, kPfTopk, kPfGemmGelu, kPfRoute, kPfMoeDown,
                kPfFfnOut, kPfGather, kPfKernels };
const char* const kPfKernelNames[kPfKernels] = {"pf_attn_rms", "pf_gemm",     "pf_tile",      "pf_ffn_in",
                                                "pf_gemm_f16w", "pf_topk",     "pf_gemm_gelu", "pf_moe_route",
                                                "pf_gemm_moe_down", "pf_ffn_out", "pf_gather"};

// Calls of at least this many tokens take the batched path (OV_MEGAKERNEL_PF_MIN).
int prefill_min_tokens() {
    static const int v = [] {
        const char* e = std::getenv("OV_MEGAKERNEL_PF_MIN");
        return e ? std::atoi(e) : 2;
    }();
    return v;
}

size_t round_up(size_t v, size_t m) {
    return (v + m - 1) / m * m;
}

// Scratch layout of one kFfn token (floats unless noted).
enum : size_t {
    kScO = 0,
    kScH1 = kScO + kHidden,
    kScX2 = kScH1 + kHidden,
    kScAct = kScX2 + kHidden,
    kScLogits = kScAct + kInter,
    kScMoeAct = kScLogits + kExperts,
    kScM = kScMoeAct + kTopK * kExpertInter,
    kScE = kScM + kHidden,
    kScTkIdx = kScE + kHidden,  // i32
    kScTkW = kScTkIdx + kTopK,
    kScSize = kScTkW + kTopK,
};

enum FfnKernel { kOProj, kIn, kMid, kMoeDown, kOut };
constexpr int kPrefillSyncTokens = 64;
const char* const kFfnKernelNames[kFfnKernels] = {"ffn_oproj", "ffn_in", "ffn_mid", "ffn_moe_down", "ffn_out"};

}  // namespace

Gemma4POCRuntime::Gemma4POCRuntime() = default;

Gemma4POCRuntime::~Gemma4POCRuntime() {
    Destroy();
}

TErrorcode Gemma4POCRuntime::Init(const IConstantParams* weights, const IPlatformParams* platform) {
    w_ = *static_cast<const Gemma4ConstantParams*>(weights);
    const auto* pp = static_cast<const Gemma4PlatformParams*>(platform);
    ctx_ = pp->context;
    dev_ = pp->deviceId;
    stream_ = pp->stream;

    cl_platform_id plat = nullptr;
    clGetDeviceInfo(dev_, CL_DEVICE_PLATFORM, sizeof(plat), &plat, nullptr);
    usmAlloc_ = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clDeviceMemAllocINTEL"));
    usmFree_ = reinterpret_cast<clMemFreeINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clMemFreeINTEL"));
    setUsmArg_ = reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL"));
    usmBlockingFree_ = reinterpret_cast<clMemBlockingFreeINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(plat, "clMemBlockingFreeINTEL"));
    if (!usmAlloc_ || !usmFree_ || !setUsmArg_) {
        std::fprintf(stderr, "[Gemma4] Intel USM extensions unavailable\n");
        return CL_INVALID_PLATFORM;
    }
    if (w_.kind == kModel) {
        Gemma4ModelRuntime::Usm u{};
        u.alloc = usmAlloc_;
        u.free = usmFree_;
        u.blocking_free = usmBlockingFree_;
        u.set_arg = setUsmArg_;
        u.memcpy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(plat, "clEnqueueMemcpyINTEL"));
        model_ = std::make_unique<Gemma4ModelRuntime>();
        return model_->Init(w_, ctx_, dev_, stream_, u);
    }
    if (w_.kind == kAttnIn)
        return InitAttn();
    if (w_.kind == kMaskFull || w_.kind == kMaskSliding)
        return InitMask();
    return InitFfn();
}

TErrorcode Gemma4POCRuntime::InitFfn() {
    cl_int err = CL_SUCCESS;
    cl_program prog = ffn_program(ctx_, dev_, &err);
    CL_OK(err, "build ffn program");
    for (int i = 0; i < kFfnKernels; ++i) {
        kFfn_[i] = clCreateKernel(prog, kFfnKernelNames[i], &err);
        CL_OK(err, kFfnKernelNames[i]);
    }

    scratch_ = usmAlloc_(ctx_, dev_, nullptr, kScSize * sizeof(float), 0, &err);
    CL_OK(err, "scratch alloc");
    auto* sc = static_cast<float*>(scratch_);

    ffn_global_sizes(kHidden, kInter, kExpertInter, kExperts, kTopK, ffnGws_);

    if (std::getenv("OV_MEGAKERNEL_DUMP")) {
        for (cl_kernel k : kFfn_) {
            cl_ulong spill = 0, slm = 0;
            char name[64] = {0};
            clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(name) - 1, name, nullptr);
            clGetKernelWorkGroupInfo(k, dev_, 0x4109 /*CL_KERNEL_SPILL_MEM_SIZE_INTEL*/, sizeof(spill), &spill, nullptr);
            clGetKernelWorkGroupInfo(k, dev_, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(slm), &slm, nullptr);
            std::fprintf(stderr, "[Gemma4] kernel %s spill=%llu slm=%llu\n", name, (unsigned long long)spill,
                         (unsigned long long)slm);
        }
    }

    auto setp = [&](cl_kernel k, cl_uint i, const void* p) { setUsmArg_(k, i, p); };
    void* const* f = w_.ffn;
    // ffn_oproj: args 0 (attention output) and 2 (its width) are per token.
    setp(kFfn_[kOProj], 1, f[kFfnOProj]);
    setp(kFfn_[kOProj], 3, sc + kScO);

    // ffn_in: arg 1 (residual) is per token.
    setp(kFfn_[kIn], 0, sc + kScO);
    cl_uint a = 2;
    for (int port : {kFfnWPostAttn, kFfnWPreFf, kFfnWRouter, kFfnWPreFf2, kFfnGate, kFfnUp, kFfnRouterW})
        setp(kFfn_[kIn], a++, f[port]);
    for (size_t off : {kScH1, kScX2, kScAct, kScLogits})
        setp(kFfn_[kIn], a++, sc + off);

    a = 0;
    for (size_t off : {kScX2, kScLogits, kScAct})
        setp(kFfn_[kMid], a++, sc + off);
    for (int port : {kFfnEGate, kFfnEUp, kFfnDown})
        setp(kFfn_[kMid], a++, f[port]);
    for (size_t off : {kScMoeAct, kScM, kScTkIdx, kScTkW})
        setp(kFfn_[kMid], a++, sc + off);

    a = 0;
    for (size_t off : {kScMoeAct, kScTkIdx, kScTkW})
        setp(kFfn_[kMoeDown], a++, sc + off);
    setp(kFfn_[kMoeDown], a++, f[kFfnEDown]);
    setp(kFfn_[kMoeDown], a++, sc + kScE);

    a = 0;
    for (size_t off : {kScH1, kScM, kScE})
        setp(kFfn_[kOut], a++, sc + off);
    for (int port : {kFfnWPostFf1, kFfnWPostFf2, kFfnWPostFf, kFfnLayerScalar})
        setp(kFfn_[kOut], a++, f[port]);
    // arg 7 (out) is per token.
    return InitPrefill();
}

TErrorcode Gemma4POCRuntime::InitPrefill() {
    for (int v = 0; v < 2; ++v) {
        cl_int err = CL_SUCCESS;
        cl_program prog = ffn_program(ctx_, dev_, &err, v ? kPfMbSmall : kPfMb);
        CL_OK(err, "build prefill program");
        for (int i = 0; i < kPfKernels; ++i) {
            kPf_[v][i] = clCreateKernel(prog, kPfKernelNames[i], &err);
            CL_OK(err, kPfKernelNames[i]);
            if (std::getenv("OV_MEGAKERNEL_DUMP")) {
                cl_ulong spill = 0;
                clGetKernelWorkGroupInfo(kPf_[v][i], dev_, 0x4109 /*CL_KERNEL_SPILL_MEM_SIZE_INTEL*/, sizeof(spill),
                                         &spill, nullptr);
                std::fprintf(stderr, "[Gemma4] kernel %s/%d spill=%llu\n", kPfKernelNames[i], v,
                             (unsigned long long)spill);
            }
        }
    }
    std::lock_guard<std::mutex> g(g_pf_mu);
    ++g_pf[ctx_].refs;
    pfShared_ = true;
    return CL_SUCCESS;
}

void* Gemma4POCRuntime::pf_buf(int id, size_t bytes) {
    std::lock_guard<std::mutex> g(g_pf_mu);
    PfScratch& s = g_pf[ctx_];
    if (s.cap[id] < bytes) {
        // Earlier layers of this call may still use the old block.
        if (s.buf[id])
            usmBlockingFree_(ctx_, s.buf[id]);
        cl_int err = CL_SUCCESS;
        const size_t cap = round_up(bytes + bytes / 4, 4096);
        s.buf[id] = usmAlloc_(ctx_, dev_, nullptr, cap, 0, &err);
        s.cap[id] = err == CL_SUCCESS && s.buf[id] ? cap : 0;
        if (!s.cap[id])
            s.buf[id] = nullptr;
    }
    return s.buf[id];
}

// Kernel k of the tile variant `v` in scope (0: kPfMb, 1: kPfMbSmall).
#define PF_ARG(k, i, x) CL_OK(clSetKernelArg(kPf_[v][k], i, sizeof(x), &(x)), kPfKernelNames[k])
#define PF_PTR(k, i, p) CL_OK(setUsmArg_(kPf_[v][k], i, p), kPfKernelNames[k])
#define PF_RUN(k, dims, gws, lws) \
    CL_OK(clEnqueueNDRangeKernel(stream_, kPf_[v][k], dims, nullptr, gws, lws, 0, nullptr, nullptr), kPfKernelNames[k])

// Variant for the dense GEMMs of a T-token call, and for its MoE.
static int pf_dense_variant(int T) {
    static const int lim = [] {
        const char* e = std::getenv("OV_MEGAKERNEL_PF_DENSE_SMALL");
        return e ? std::atoi(e) : 32;
    }();
    return T <= lim ? 1 : 0;
}
static int pf_moe_variant(int T) {
    // Few rows per expert: 8-row tiles waste far less XMX work than 64-row ones.
    static const int lim = [] {
        const char* e = std::getenv("OV_MEGAKERNEL_PF_MOE_SMALL");
        return e ? std::atoi(e) : 256;
    }();
    return T <= lim ? 1 : 0;
}
static int pf_mt(int v) {
    return 8 * (v ? kPfMbSmall : kPfMb);
}

TErrorcode Gemma4POCRuntime::PrefillAttn(const Gemma4RuntimeParams& p) {
    const int T = p.tokens, R = w_.attn_rows, H = kHidden;
    const int v = pf_dense_variant(T);
    const size_t mtiles = (T + pf_mt(v) - 1) / pf_mt(v);
    void* at = pf_buf(kPfAT, static_cast<size_t>(T) * H * sizeof(uint16_t));
    void* raw = pf_buf(kPfRaw, static_cast<size_t>(T) * R * sizeof(float));
    if (!at || !raw)
        return CL_OUT_OF_RESOURCES;
    void* const* a = w_.attn;

    PF_PTR(kPfAttnRms, 0, p.in);
    PF_PTR(kPfAttnRms, 1, a[kAttnWIn]);
    PF_PTR(kPfAttnRms, 2, at);
    const size_t rms_gws = kFfnLws * T, lws = kFfnLws;
    PF_RUN(kPfAttnRms, 1, &rms_gws, &lws);

    const int nrb = R / 16;
    PF_PTR(kPfGemm, 0, at);
    PF_ARG(kPfGemm, 1, H);
    PF_PTR(kPfGemm, 2, a[kAttnQkv]);
    PF_ARG(kPfGemm, 3, nrb);
    PF_ARG(kPfGemm, 4, T);
    PF_PTR(kPfGemm, 5, raw);
    PF_ARG(kPfGemm, 6, R);
    const size_t g_gws[2] = {round_up(nrb / 2, kPfNsg) / kPfNsg * kPfLws, mtiles};
    const size_t g_lws[2] = {kPfLws, 1};
    PF_RUN(kPfGemm, 2, g_gws, g_lws);

    const int t0 = 0;
    setUsmArg_(kAttn_[1], 0, raw);
    setUsmArg_(kAttn_[1], 4, p.in2);
    setUsmArg_(kAttn_[1], 6, p.out);
    CL_OK(clSetKernelArg(kAttn_[1], 7, sizeof(int), &t0), "set t0");
    CL_OK(clSetKernelArg(kAttn_[1], 8, sizeof(int), &p.s), "set S");
    CL_OK(clSetKernelArg(kAttn_[1], 9, sizeof(int), &R), "set rstride");
    const size_t post_gws[2] = {attnGws_[1], static_cast<size_t>(T)};
    const size_t post_lws[2] = {kFfnLws, 1};
    CL_OK(clEnqueueNDRangeKernel(stream_, kAttn_[1], 2, nullptr, post_gws, post_lws, 0, nullptr, nullptr),
          "attn_post");
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::PrefillFfn(const Gemma4RuntimeParams& p) {
    const int T = p.tokens, H = kHidden, KA = p.ka, I = kInter, IE = kExpertInter, E = kExperts;
    const int dv = pf_dense_variant(T), mv = pf_moe_variant(T);
    const size_t mtiles = (T + pf_mt(dv) - 1) / pf_mt(dv);
    const size_t emt = pf_mt(mv);
    const size_t etiles = (static_cast<size_t>(T) * kTopK + emt - 1) / emt + E;  // bound on MoE tiles
    const size_t t = T;
    void* at = pf_buf(kPfAT, t * std::max(KA, H) * 2);
    void* o = pf_buf(kPfO, t * H * 4);
    void* h1 = pf_buf(kPfH1, t * H * 4);
    void* xd = pf_buf(kPfXD, t * H * 2);
    void* xr = pf_buf(kPfXR, t * H * 2);
    void* x2 = pf_buf(kPfX2, t * H * 2);
    void* lg = pf_buf(kPfLog, kPfRouterKs * t * E * 4);
    void* tki = pf_buf(kPfTkI, t * kTopK * 4);
    void* tkw = pf_buf(kPfTkW, t * kTopK * 4);
    void* act = pf_buf(kPfAct, t * I * 2);
    void* md = pf_buf(kPfMD, t * H * 4);
    void* hdr = pf_buf(kPfHdr, 64);
    void* texp = pf_buf(kPfTExp, etiles * 4);
    void* pair = pf_buf(kPfPair, etiles * emt * 4);
    void* xe = pf_buf(kPfXE, etiles * emt * H * 2);
    void* ae = pf_buf(kPfAE, etiles * emt * IE * 2);
    void* ys = pf_buf(kPfYS, t * kTopK * H * 2);
    for (void* b : {at, o, h1, xd, xr, x2, lg, tki, tkw, act, md, hdr, texp, pair, xe, ae, ys})
        if (!b)
            return CL_OUT_OF_RESOURCES;
    void* const* f = w_.ffn;
    void* const null = nullptr;
    const size_t lws = kFfnLws, tok_gws = kFfnLws * T;
    const size_t g_lws[2] = {kPfLws, 1};
    const size_t tile_lws[2] = {16, 16};

    auto gemm = [&](int v, void* a, int K, void* w, int N, void* out) -> TErrorcode {
        const int nrb = N / 16;
        PF_PTR(kPfGemm, 0, a);
        PF_ARG(kPfGemm, 1, K);
        PF_PTR(kPfGemm, 2, w);
        PF_ARG(kPfGemm, 3, nrb);
        PF_ARG(kPfGemm, 4, T);
        PF_PTR(kPfGemm, 5, out);
        PF_ARG(kPfGemm, 6, N);
        const size_t gws[2] = {round_up(nrb / 2, kPfNsg) / kPfNsg * kPfLws, mtiles};
        PF_RUN(kPfGemm, 2, gws, g_lws);
        return CL_SUCCESS;
    };
    auto gelu = [&](int v, void* a, int K, void* wg, void* wu, int N, void* hd, void* te, size_t tiles,
                    void* out) -> TErrorcode {
        const int nrb = N / 16;
        PF_PTR(kPfGemmGelu, 0, a);
        PF_ARG(kPfGemmGelu, 1, K);
        PF_PTR(kPfGemmGelu, 2, wg);
        PF_PTR(kPfGemmGelu, 3, wu);
        PF_ARG(kPfGemmGelu, 4, nrb);
        PF_ARG(kPfGemmGelu, 5, T);
        PF_PTR(kPfGemmGelu, 6, hd);
        PF_PTR(kPfGemmGelu, 7, te);
        PF_PTR(kPfGemmGelu, 8, out);
        const size_t gws[2] = {round_up(nrb, kPfNsg) / kPfNsg * kPfLws, tiles};
        PF_RUN(kPfGemmGelu, 2, gws, g_lws);
        return CL_SUCCESS;
    };
    auto tile = [&](int v, void* src, int K, void* out) -> TErrorcode {
        PF_PTR(kPfTile, 0, src);
        PF_ARG(kPfTile, 1, K);
        PF_ARG(kPfTile, 2, T);
        PF_PTR(kPfTile, 3, out);
        const size_t gws[2] = {round_up(K / 16, 16), round_up(t, 16)};
        PF_RUN(kPfTile, 2, gws, tile_lws);
        return CL_SUCCESS;
    };
#define PF_DO(x)                     \
    do {                             \
        const TErrorcode e_ = (x);   \
        if (e_ != CL_SUCCESS)        \
            return e_;               \
    } while (0)

    // o = o_proj(attn)
    PF_DO(tile(dv, p.in, KA, at));
    PF_DO(gemm(dv, at, KA, f[kFfnOProj], H, o));

    int v = 0;
    // h1, the three normalised activations
    {
        cl_uint a = 0;
        for (void* x : {o, p.in2, f[kFfnWPostAttn], f[kFfnWPreFf], f[kFfnWRouter], f[kFfnWPreFf2], h1, xd, xr, x2})
            PF_PTR(kPfFfnIn, a++, x);
        PF_RUN(kPfFfnIn, 1, &tok_gws, &lws);
    }

    // router logits (K-split partials), top-k
    {
        PF_PTR(kPfGemmF16w, 0, xr);
        PF_ARG(kPfGemmF16w, 1, H);
        PF_PTR(kPfGemmF16w, 2, f[kFfnRouterW]);
        PF_ARG(kPfGemmF16w, 3, E);
        PF_ARG(kPfGemmF16w, 4, T);
        PF_PTR(kPfGemmF16w, 5, lg);
        const size_t gws[2] = {kPfRouterKs * kPfLws, (t + kPfMt - 1) / kPfMt};
        PF_RUN(kPfGemmF16w, 2, gws, g_lws);
        const int ks = kPfRouterKs;
        PF_PTR(kPfTopk, 0, lg);
        PF_ARG(kPfTopk, 1, T);
        PF_ARG(kPfTopk, 2, ks);
        PF_PTR(kPfTopk, 3, tki);
        PF_PTR(kPfTopk, 4, tkw);
        const size_t tk_gws = round_up(t, 16) / 16 * kFfnLws;
        PF_RUN(kPfTopk, 1, &tk_gws, &lws);
    }

    // dense MLP
    PF_DO(gelu(dv, xd, H, f[kFfnGate], f[kFfnUp], I, null, null, mtiles, act));
    PF_DO(gemm(dv, act, I, f[kFfnDown], H, md));

    // MoE: bucket pairs per expert, gather, grouped gate/up, grouped down + scatter
    v = mv;
    {
        PF_PTR(kPfRoute, 0, tki);
        PF_ARG(kPfRoute, 1, T);
        PF_PTR(kPfRoute, 2, hdr);
        PF_PTR(kPfRoute, 3, texp);
        PF_PTR(kPfRoute, 4, pair);
        const size_t r_gws = 1024;
        PF_RUN(kPfRoute, 1, &r_gws, &r_gws);
    }
    {
        PF_PTR(kPfGather, 0, x2);
        PF_ARG(kPfGather, 1, H);
        PF_PTR(kPfGather, 2, hdr);
        PF_PTR(kPfGather, 3, pair);
        PF_PTR(kPfGather, 4, xe);
        const size_t lw[2] = {static_cast<size_t>(H) * 2 / 512 * 16, 1};
        const size_t gws[2] = {lw[0], etiles * emt};
        PF_RUN(kPfGather, 2, gws, lw);
    }
    PF_DO(gelu(mv, xe, H, f[kFfnEGate], f[kFfnEUp], IE, hdr, texp, etiles, ae));
    {
        const int nrb = H / 16;
        cl_uint a = 0;
        PF_PTR(kPfMoeDown, a++, ae);
        PF_ARG(kPfMoeDown, a++, IE);
        PF_PTR(kPfMoeDown, a++, f[kFfnEDown]);
        PF_ARG(kPfMoeDown, a++, nrb);
        for (void* x : {hdr, texp, pair, tkw, ys})
            PF_PTR(kPfMoeDown, a++, x);
        const size_t gws[2] = {round_up(nrb / 2, kPfNsg) / kPfNsg * kPfLws, etiles};
        PF_RUN(kPfMoeDown, 2, gws, g_lws);
    }

    // out
    v = 0;
    {
        cl_uint a = 0;
        for (void* x : {h1, md, ys, f[kFfnWPostFf1], f[kFfnWPostFf2], f[kFfnWPostFf], f[kFfnLayerScalar], p.out})
            PF_PTR(kPfFfnOut, a++, x);
        PF_RUN(kPfFfnOut, 1, &tok_gws, &lws);
    }
#undef PF_DO
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::ExecuteFfn(const Gemma4RuntimeParams& p) {
    if (p.tokens >= prefill_min_tokens())
        return PrefillFfn(p);
    const size_t lws = kFfnLws;
    const size_t row = kHidden * sizeof(uint16_t);
    const size_t attn_row = static_cast<size_t>(p.ka) * sizeof(uint16_t);
    CL_OK(clSetKernelArg(kFfn_[kOProj], 2, sizeof(int), &p.ka), "set o_proj width");
    // The block is token-local, so prefill simply repeats the decode launches.
    for (int t = 0; t < p.tokens; ++t) {
        setUsmArg_(kFfn_[kOProj], 0, static_cast<char*>(p.in) + t * attn_row);
        setUsmArg_(kFfn_[kIn], 1, static_cast<char*>(p.in2) + t * row);
        setUsmArg_(kFfn_[kOut], 7, static_cast<char*>(p.out) + t * row);
        for (int k = 0; k < kFfnKernels; ++k)
            CL_OK(clEnqueueNDRangeKernel(stream_, kFfn_[k], 1, nullptr, &ffnGws_[k], &lws, 0, nullptr, nullptr),
                  kFfnKernelNames[k]);
        // A long prefill would otherwise enqueue ~10^6 launches without ever
        // blocking and exhaust the driver's command storage (CL_OUT_OF_RESOURCES
        // at ~5k tokens), so bound the work in flight.
        if (t % kPrefillSyncTokens == kPrefillSyncTokens - 1)
            CL_OK(clFinish(stream_), "prefill throttle");
    }
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::Execute(const IRuntimeParams* io) {
    const auto* p = static_cast<const Gemma4RuntimeParams*>(io);
    if (!setUsmArg_) {
        std::fprintf(stderr, "[Gemma4] runtime not initialised\n");
        return CL_INVALID_KERNEL;
    }
    if (w_.kind == kModel)
        return model_ ? model_->Execute(*p) : CL_INVALID_KERNEL;
    if (w_.kind == kFfn)
        return kFfn_[kOut] ? ExecuteFfn(*p) : CL_INVALID_KERNEL;
    if (w_.kind == kAttnIn)
        return kAttn_[1] ? ExecuteAttn(*p) : CL_INVALID_KERNEL;
    if (w_.kind == kMaskFull || w_.kind == kMaskSliding)
        return kMask_[1] ? ExecuteMask(*p) : CL_INVALID_KERNEL;
    return CL_INVALID_KERNEL;
}

TErrorcode Gemma4POCRuntime::InitMask() {
    cl_int err = CL_SUCCESS;
    cl_program prog = ffn_program(ctx_, dev_, &err);
    CL_OK(err, "build ffn program");
    kMask_[0] = clCreateKernel(prog, "mask_groups", &err);
    CL_OK(err, "mask_groups");
    kMask_[1] = clCreateKernel(prog, "mask_fill", &err);
    CL_OK(err, "mask_fill");
    setUsmArg_(kMask_[0], 1, w_.mask_meta);
    setUsmArg_(kMask_[1], 1, w_.mask_meta);
    const int sliding = w_.kind == kMaskSliding;
    CL_OK(clSetKernelArg(kMask_[1], 5, sizeof(int), &sliding), "set sliding");
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::ExecuteMask(const Gemma4RuntimeParams& p) {
    const int B = p.tokens, L = p.ka, S = p.s;
    if (B <= 0 || L <= 0 || S <= 0 || S > L)
        return CL_INVALID_VALUE;
    if (w_.kind == kMaskSliding) {
        const size_t need = static_cast<size_t>(B) * L;
        if (need > maskGrpCap_) {
            if (scratch_ && usmBlockingFree_)
                usmBlockingFree_(ctx_, scratch_);
            cl_int err = CL_SUCCESS;
            maskGrpCap_ = need * 2;
            scratch_ = usmAlloc_(ctx_, dev_, nullptr, maskGrpCap_ * sizeof(int), 0, &err);
            CL_OK(err, "mask scratch alloc");
        }
        setUsmArg_(kMask_[0], 0, p.in2);
        CL_OK(clSetKernelArg(kMask_[0], 2, sizeof(int), &S), "set S");
        CL_OK(clSetKernelArg(kMask_[0], 3, sizeof(int), &L), "set L");
        setUsmArg_(kMask_[0], 4, scratch_);
        const size_t gws = 256 * static_cast<size_t>(B), lws = 256;
        CL_OK(clEnqueueNDRangeKernel(stream_, kMask_[0], 1, nullptr, &gws, &lws, 0, nullptr, nullptr),
              "mask_groups");
    }
    setUsmArg_(kMask_[1], 0, p.in);
    setUsmArg_(kMask_[1], 2, scratch_);
    CL_OK(clSetKernelArg(kMask_[1], 3, sizeof(int), &S), "set S");
    CL_OK(clSetKernelArg(kMask_[1], 4, sizeof(int), &L), "set L");
    setUsmArg_(kMask_[1], 6, p.out);
    const size_t gws[3] = {(static_cast<size_t>(L) + 63) / 64 * 64, static_cast<size_t>(S), static_cast<size_t>(B)};
    const size_t lws[3] = {64, 1, 1};
    CL_OK(clEnqueueNDRangeKernel(stream_, kMask_[1], 3, nullptr, gws, lws, 0, nullptr, nullptr), "mask_fill");
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::InitAttn() {
    if (w_.attn_rows <= 0 || w_.attn_rows % kAttnQkvRowsPerWg || w_.attn_rows > kAttnMaxRows) {
        std::fprintf(stderr, "[Gemma4] bad q|k|v row count %d\n", w_.attn_rows);
        return CL_INVALID_VALUE;
    }
    cl_int err = CL_SUCCESS;
    cl_program prog = ffn_program(ctx_, dev_, &err);
    CL_OK(err, "build ffn program");
    kAttn_[0] = clCreateKernel(prog, "attn_qkv", &err);
    CL_OK(err, "attn_qkv");
    kAttn_[1] = clCreateKernel(prog, "attn_post", &err);
    CL_OK(err, "attn_post");

    scratch_ = usmAlloc_(ctx_, dev_, nullptr, w_.attn_rows * sizeof(float), 0, &err);
    CL_OK(err, "scratch alloc");
    attnGws_[0] = kFfnLws * (w_.attn_rows / kAttnQkvRowsPerWg);
    attnGws_[1] = kFfnLws * (kAttnMaxHeads / 16);

    void* const* a = w_.attn;
    setUsmArg_(kAttn_[0], 1, a[kAttnWIn]);
    setUsmArg_(kAttn_[0], 2, a[kAttnQkv]);
    setUsmArg_(kAttn_[0], 3, scratch_);
    setUsmArg_(kAttn_[1], 0, scratch_);
    setUsmArg_(kAttn_[1], 1, a[kAttnWQ]);
    setUsmArg_(kAttn_[1], 2, a[kAttnWK]);
    setUsmArg_(kAttn_[1], 3, a[kAttnMeta]);
    setUsmArg_(kAttn_[1], 5, a[kAttnInvFreq]);
    return InitPrefill();
}

TErrorcode Gemma4POCRuntime::ExecuteAttn(const Gemma4RuntimeParams& p) {
    if (p.tokens >= prefill_min_tokens())
        return PrefillAttn(p);
    const size_t lws = kFfnLws;
    const size_t row = kHidden * sizeof(uint16_t);
    const int rstride = 0;
    setUsmArg_(kAttn_[1], 0, scratch_);
    setUsmArg_(kAttn_[1], 4, p.in2);
    setUsmArg_(kAttn_[1], 6, p.out);
    CL_OK(clSetKernelArg(kAttn_[1], 8, sizeof(int), &p.s), "set S");
    CL_OK(clSetKernelArg(kAttn_[1], 9, sizeof(int), &rstride), "set rstride");
    for (int t = 0; t < p.tokens; ++t) {
        setUsmArg_(kAttn_[0], 0, static_cast<char*>(p.in) + t * row);
        CL_OK(clSetKernelArg(kAttn_[1], 7, sizeof(int), &t), "set token");
        for (int k = 0; k < 2; ++k)
            CL_OK(clEnqueueNDRangeKernel(stream_, kAttn_[k], 1, nullptr, &attnGws_[k], &lws, 0, nullptr, nullptr),
                  k ? "attn_post" : "attn_qkv");
        if (t % kPrefillSyncTokens == kPrefillSyncTokens - 1)
            CL_OK(clFinish(stream_), "prefill throttle");
    }
    return CL_SUCCESS;
}

TErrorcode Gemma4POCRuntime::Destroy() {
    model_.reset();
    for (cl_kernel& k : kFfn_) {
        if (k)
            clReleaseKernel(k);
        k = nullptr;
    }
    for (cl_kernel& k : kAttn_) {
        if (k)
            clReleaseKernel(k);
        k = nullptr;
    }
    for (cl_kernel& k : kMask_) {
        if (k)
            clReleaseKernel(k);
        k = nullptr;
    }
    for (auto& ks : kPf_) {
        for (cl_kernel& k : ks) {
            if (k)
                clReleaseKernel(k);
            k = nullptr;
        }
    }
    if (pfShared_) {
        std::lock_guard<std::mutex> g(g_pf_mu);
        PfScratch& s = g_pf[ctx_];
        if (--s.refs == 0) {
            for (void* b : s.buf)
                if (b && usmBlockingFree_)
                    usmBlockingFree_(ctx_, b);
            g_pf.erase(ctx_);
        }
        pfShared_ = false;
    }
    // In-flight kernels may still read the scratch; the blocking free waits for
    // them without touching the queue, which may already be gone.
    if (scratch_ && usmBlockingFree_)
        usmBlockingFree_(ctx_, scratch_);
    scratch_ = nullptr;
    return CL_SUCCESS;
}

}  // namespace mk

namespace mk {
namespace {

#define MK_OK(expr, what)                                                       \
    do {                                                                        \
        cl_int _e = (expr);                                                     \
        if (_e != CL_SUCCESS) {                                                 \
            std::fprintf(stderr, "[Gemma4Model] %s failed: %d\n", what, _e);    \
            return _e;                                                          \
        }                                                                       \
    } while (0)

constexpr int kLws = 256;
constexpr int kRMax = kAttnMaxRows;
constexpr int kPhasesPerLayer = 5;  // NPH in kModelSource

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    return v ? std::atoi(v) : def;
}

cl_int set_embeds_arg(const Gemma4ModelRuntime::Usm& usm, cl_kernel k, cl_uint i, const Gemma4RuntimeParams& p) {
    return p.in_mem ? clSetKernelArg(k, i, sizeof(cl_mem), &p.in_mem) : usm.set_arg(k, i, p.in);
}

// Device-side table layouts; must match LayerTab / ModelTab in kModelSource.
struct LayerTabH {
    const void *w_in, *qkv, *gq, *gk, *invf, *oproj, *w_pa, *w_pf, *w_r, *w_pf2, *gate, *up, *down, *rw, *egate,
        *eup, *edown, *w_pf1, *w_pf2p, *w_post, *ls;
    void *kc, *vc;
    int32_t R, HQ, HK, D, RN, q_off, k_off, v_off, half_c, pos_f16, sliding, pad;
};
static_assert(sizeof(LayerTabH) == 23 * 8 + 12 * 4, "LayerTab layout");

struct ModelTabH {
    void *xbuf, *raw, *att, *o, *h1, *gu, *lg, *egu, *m, *e, *part, *sync, *tick, *err;
    const void *w_final, *lm, *lm_s, *lm_z;
    float cap_div, cap_mul, attn_scale, window, causal_off, p_mul, p_add, pad0;
    int32_t pos_stride, am_stride, kv_cap, pad1;
    void* prof;
    void* kmask;
    void* qs;
};
static_assert(sizeof(ModelTabH) == 21 * 8 + 12 * 4, "ModelTab layout");

enum Scratch {
    kSXbuf, kSRaw, kSAtt, kSO, kSH1, kSGu, kSLg, kSEgu, kSM, kSE, kSPart, kSSync, kSTick, kSErr, kSQs, kSCount
};

cl_program build_model_program(cl_context ctx, cl_device_id dev, const std::string& opts, cl_int* err) {
    const char* srcs[2] = {kFfnSource, kModelSource};
    cl_program prog = clCreateProgramWithSource(ctx, 2, srcs, nullptr, err);
    if (*err != CL_SUCCESS)
        return nullptr;
    *err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    size_t n = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::vector<char> log(n + 1, 0);
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    if (*err != CL_SUCCESS || std::getenv("OV_MEGAKERNEL_DUMP"))
        std::fprintf(stderr, "[Gemma4Model] build %s:\n%s\n", *err == CL_SUCCESS ? "log" : "FAILED", log.data());
    if (*err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return nullptr;
    }
    return prog;
}

}  // namespace

Gemma4ModelRuntime::Gemma4ModelRuntime() = default;
Gemma4ModelRuntime::~Gemma4ModelRuntime() {
    Destroy();
}

void* Gemma4ModelRuntime::Alloc(size_t bytes) {
    cl_int err = CL_SUCCESS;
    void* p = usm_.alloc(ctx_, dev_, nullptr, std::max<size_t>(bytes, 64), 0, &err);
    return err == CL_SUCCESS ? p : nullptr;
}

void* Gemma4ModelRuntime::Grow(void*& buf, size_t& cap, size_t bytes) {
    if (cap >= bytes && buf)
        return buf;
    if (buf)
        usm_.blocking_free(ctx_, buf);
    cap = round_up(bytes + bytes / 4, 4096);
    buf = Alloc(cap);
    if (!buf)
        cap = 0;
    return buf;
}

int Gemma4ModelRuntime::Init(const Gemma4ConstantParams& w, cl_context ctx, cl_device_id dev, cl_command_queue q,
                             const Usm& usm) {
    ctx_ = ctx;
    dev_ = dev;
    q_ = q;
    usm_ = usm;
    w_ = w;
    nl_ = w.layers;
    if (nl_ <= 0 || nl_ > kModelMaxLayers || !usm_.memcpy || !usm_.blocking_free) {
        std::fprintf(stderr, "[Gemma4Model] bad model description (%d layers)\n", nl_);
        return CL_INVALID_VALUE;
    }

    // Scalars live in device constants; read them back once.
    std::vector<float> meta(kMetaLayer0 + nl_);
    float mm[16];
    MK_OK(usm_.memcpy(q_, CL_TRUE, meta.data(), w.model[kModelMeta], meta.size() * 4, 0, nullptr, nullptr),
          "read model meta");
    MK_OK(usm_.memcpy(q_, CL_TRUE, mm, w.model[kModelMaskMeta], sizeof(mm), 0, nullptr, nullptr), "read mask meta");
    vocab_ = static_cast<int>(meta[kMetaVocab]);
    cap_div_ = meta[kMetaCapDiv];
    cap_mul_ = meta[kMetaCapMul];
    scale_ = meta[kMetaAttnScale];
    last_only_ = meta[kMetaLastOnly] != 0.0f;
    p_mul_ = mm[2];
    p_add_ = mm[3];
    window_ = mm[4];
    causal_off_ = mm[5];
    am_stride_ = static_cast<int>(mm[8]);

    layers_.resize(nl_);
    int hq_max = 0, hq_min = 1 << 30, d_max = 0, hm_max = 0;
    for (int l = 0; l < nl_; ++l) {
        int am[16];
        MK_OK(usm_.memcpy(q_, CL_TRUE, am, w.layer[l][kAttnMeta - kAttnWIn], sizeof(am), 0, nullptr, nullptr),
              "read attention meta");
        Layer& L = layers_[l];
        L.HQ = am[0];
        L.HK = am[1];
        L.D = am[2];
        L.RN = am[3];
        L.k_off = am[4];
        L.v_off = am[5];
        L.q_off = am[6];
        pos_stride_ = am[7];
        L.pos_f16 = am[8];
        L.half_c = am[9] / 2;
        L.R = w.layer_qkv_rows[l];
        L.sliding = meta[kMetaLayer0 + l] != 0.0f;
        const int gps = (L.D == 256 || L.D == 512) ? 512 / L.D : 0;
        const int G = L.HK ? L.HQ / L.HK : 0;
        const int qsplit = gps ? G / gps : 0;
        // mk_decode instantiates the attention for full rotary heads of 256 / 512.
        if (!gps || !G || L.HQ % L.HK || G % gps || !qsplit || 16 % qsplit || L.HK > 8 || L.HQ > 16 || G > 8 ||
            L.RN != L.D || L.R > kRMax || L.R % 16 || L.half_c <= 0) {
            std::fprintf(stderr, "[Gemma4Model] layer %d: unsupported attention HQ=%d HK=%d D=%d RN=%d R=%d\n", l,
                         L.HQ, L.HK, L.D, L.RN, L.R);
            return CL_INVALID_VALUE;
        }
        hq_max = std::max(hq_max, L.HQ);
        hq_min = std::min(hq_min, L.HQ);
        d_max = std::max(d_max, L.D);
        hm_max = std::max(hm_max, (L.HQ + 2 * L.HK) * L.D);
        int& mk = max_kv_heads_[L.sliding ? 0 : 1];
        mk = std::max(mk, L.HK);
    }

    // 256 registers per thread (the attention phase spills with 128) leave
    // room for two 256-wide work-groups per Xe-core.
    const bool big_grf = env_int("OV_MEGAKERNEL_GRF", 256) == 256;
    cl_uint cus = 0;
    clGetDeviceInfo(dev_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
    const int xe_cores = std::max(1, static_cast<int>(cus) / 8);
    nwg_ = env_int("OV_MEGAKERNEL_NWG", xe_cores * (big_grf ? 2 : 4));
    if (nwg_ < kExperts / 16 + 1 || nwg_ < max_kv_heads_[0] || nwg_ < max_kv_heads_[1]) {
        std::fprintf(stderr, "[Gemma4Model] OV_MEGAKERNEL_NWG=%d too small\n", nwg_);
        return CL_INVALID_VALUE;
    }

    char opts[512];
    std::snprintf(opts, sizeof(opts),
                  "-DH_=%d -DI_=%d -DIE_=%d -DE_=%d -DTOPK_=%d -DEPS_=1e-6f -DNL_=%d -DNWG=%d -DV_=%d %s",
                  kHidden, kInter, kExpertInter, kExperts, kTopK, nl_, nwg_, vocab_, kFfnBuildOpts);
    std::string o = opts;
    if (big_grf)
        o += " -cl-intel-256-GRF-per-thread";
    prof_on_ = env_int("OV_MEGAKERNEL_PROF", 0) != 0;
    debug_cmp_ = env_int("OV_MEGAKERNEL_DEBUG_CMP", 0) != 0;
    if (prof_on_)
        o += " -DMK_PROF";
    if (prof_on_ && env_int("OV_MEGAKERNEL_PROF", 0) == 3) {
        prof_total_ = true;
        o += " -DMK_PROF_TOTAL";
    }
    else if (debug_cmp_)
        o += " -DMK_DEBUG";
    if (const char* extra = std::getenv("OV_MEGAKERNEL_CL_OPTS"))
        o += std::string(" ") + extra;
    cl_int err = CL_SUCCESS;
    prog_ = build_model_program(ctx_, dev_, o, &err);
    MK_OK(err, "build model program");
    struct {
        cl_kernel* k;
        const char* name;
    } ks[] = {{&kDecode_, "mk_decode"},     {&kCvt_, "mk_cvt_embeds"}, {&kKvAppend_, "mk_kv_append"},
              {&kMaskRange_, "mk_mask_range"}, {&kPfAttn_, "mk_pf_attn"}, {&kPfFa_, "mk_pf_fa"},  {&kLmRows_, "mk_lm_head_rows"}};
    for (auto& k : ks) {
        *k.k = clCreateKernel(prog_, k.name, &err);
        MK_OK(err, k.name);
        if (std::getenv("OV_MEGAKERNEL_DUMP")) {
            cl_ulong spill = 0, slm = 0, priv = 0;
            clGetKernelWorkGroupInfo(*k.k, dev_, 0x4109 /*CL_KERNEL_SPILL_MEM_SIZE_INTEL*/, sizeof(spill), &spill,
                                     nullptr);
            clGetKernelWorkGroupInfo(*k.k, dev_, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(slm), &slm, nullptr);
            clGetKernelWorkGroupInfo(*k.k, dev_, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(priv), &priv, nullptr);
            std::fprintf(stderr, "[Gemma4Model] kernel %s spill=%llu slm=%llu private=%llu\n", k.name,
                         (unsigned long long)spill, (unsigned long long)slm, (unsigned long long)priv);
        }
    }
    // mk_decode reaches every weight through the tables.
    cl_bool on = CL_TRUE;
    clSetKernelExecInfo(kDecode_, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(on), &on);
    clSetKernelExecInfo(kDecode_, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof(on), &on);
    clSetKernelExecInfo(kDecode_, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, sizeof(on), &on);
    clSetKernelExecInfo(kLmRows_, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(on), &on);

    // Decode scratch.
    const int p_max = nwg_ * 16 * 2 / std::max(1, hq_min);
    const size_t sizes[kSCount] = {
        size_t(kHidden) * 2,
        size_t(kRMax) * 8,  // fixed-point accumulators are 64-bit
        size_t(hq_max) * d_max * 2,
        size_t(kHidden) * 8,
        size_t(kHidden) * 4,
        size_t(2) * kInter * 8,
        size_t(kExperts) * 8,
        size_t(kTopK) * 2 * kExpertInter * 8,
        size_t(kHidden) * 8,
        size_t(kHidden) * 8,
        size_t(hq_max) * p_max * (d_max + 16) * 4,
        size_t(nl_ * kPhasesPerLayer + 1 + nl_ + nl_ * 4) * 4,  // barriers, routing, run_jobs_pool
        size_t(nl_) * 8 * 4,
        4,
        size_t(nl_) * 16 * 512 * 2,
    };
    size_t total = 0;
    for (int i = 0; i < kSCount; ++i) {
        off_[i] = total;
        total += round_up(sizes[i], 256);
    }
    scratch_ = Alloc(total);
    if (!scratch_)
        return CL_OUT_OF_RESOURCES;
    {
        std::vector<uint8_t> zeros(total, 0);
        MK_OK(usm_.memcpy(q_, CL_TRUE, scratch_, zeros.data(), total, 0, nullptr, nullptr), "zero scratch");
    }
    tab_ = Alloc(sizeof(ModelTabH));
    ltab_ = Alloc(sizeof(LayerTabH) * nl_);
    if (!tab_ || !ltab_)
        return CL_OUT_OF_RESOURCES;
    if (debug_cmp_ && !prof_on_) {
        prof_ = Alloc(size_t(nl_ + 1) * kHidden * 2 + size_t(nl_) * 8192 * 2);
        if (!prof_)
            return CL_OUT_OF_RESOURCES;
    }
    if (prof_on_) {
        prof_ = Alloc((size_t(nl_ * kPhasesPerLayer + 2) * nwg_ * 5 + size_t(nl_) * 16) * 8);
        if (!prof_)
            return CL_OUT_OF_RESOURCES;
    }

    const int rc = EnsureCapacity(env_int("OV_MEGAKERNEL_KV_CAP", 4096));
    if (rc != CL_SUCCESS)
        return rc;

    MK_OK(usm_.set_arg(kDecode_, 0, tab_), "mk_decode tab");
    MK_OK(usm_.set_arg(kDecode_, 1, ltab_), "mk_decode layers");

    // Prompt path: per-layer runtimes over the same device weights.
    Gemma4PlatformParams pp{};
    pp.deviceId = dev_;
    pp.context = ctx_;
    pp.stream = q_;
    for (int l = 0; l < nl_; ++l) {
        Gemma4ConstantParams a{};
        a.kind = kAttnIn;
        for (int p = 0; p < kModelAttnPorts; ++p)
            a.attn[kAttnWIn + p] = w.layer[l][p];
        a.attn_rows = layers_[l].R;
        attn_rt_.emplace_back(new Gemma4POCRuntime());
        MK_OK(attn_rt_.back()->Init(&a, &pp), "attention runtime");
        Gemma4ConstantParams f{};
        f.kind = kFfn;
        for (int p = 0; p < kModelFfnPorts; ++p)
            f.ffn[kFfnWPostAttn + p] = w.layer[l][kModelAttnPorts + p];
        ffn_rt_.emplace_back(new Gemma4POCRuntime());
        MK_OK(ffn_rt_.back()->Init(&f, &pp), "ffn runtime");
    }
    for (int kind : {kMaskFull, kMaskSliding}) {
        Gemma4ConstantParams m{};
        m.kind = kind;
        m.mask_meta = w.model[kModelMaskMeta];
        mask_rt_.emplace_back(new Gemma4POCRuntime());
        MK_OK(mask_rt_.back()->Init(&m, &pp), "mask runtime");
    }
    (void)hm_max;
    if (std::getenv("OV_MEGAKERNEL_DUMP"))
        std::fprintf(stderr,
                     "[Gemma4Model] %d layers, V=%d, NWG=%d, softcap %g/%g, scale %g, window %g, last_only %d, "
                     "pos stride %d, am stride %d\n",
                     nl_, vocab_, nwg_, cap_div_, cap_mul_, scale_, window_, last_only_, pos_stride_, am_stride_);
    return CL_SUCCESS;
}

int Gemma4ModelRuntime::UploadTables() {
    hltab_.assign(sizeof(LayerTabH) * nl_, 0);
    auto* lt = reinterpret_cast<LayerTabH*>(hltab_.data());
    for (int l = 0; l < nl_; ++l) {
        void* const* a = w_.layer[l];
        void* const* f = w_.layer[l] + kModelAttnPorts - kFfnWPostAttn;  // indexed by FfnPort
        const Layer& L = layers_[l];
        LayerTabH& t = lt[l];
        t.w_in = a[kAttnWIn - kAttnWIn];
        t.qkv = a[kAttnQkv - kAttnWIn];
        t.gq = a[kAttnWQ - kAttnWIn];
        t.gk = a[kAttnWK - kAttnWIn];
        t.invf = a[kAttnInvFreq - kAttnWIn];
        t.oproj = f[kFfnOProj];
        t.w_pa = f[kFfnWPostAttn];
        t.w_pf = f[kFfnWPreFf];
        t.w_r = f[kFfnWRouter];
        t.w_pf2 = f[kFfnWPreFf2];
        t.gate = f[kFfnGate];
        t.up = f[kFfnUp];
        t.down = f[kFfnDown];
        t.rw = f[kFfnRouterW];
        t.egate = f[kFfnEGate];
        t.eup = f[kFfnEUp];
        t.edown = f[kFfnEDown];
        t.w_pf1 = f[kFfnWPostFf1];
        t.w_pf2p = f[kFfnWPostFf2];
        t.w_post = f[kFfnWPostFf];
        t.ls = f[kFfnLayerScalar];
        t.kc = L.kc;
        t.vc = L.vc;
        t.R = L.R;
        t.HQ = L.HQ;
        t.HK = L.HK;
        t.D = L.D;
        t.RN = L.RN;
        t.q_off = L.q_off;
        t.k_off = L.k_off;
        t.v_off = L.v_off;
        t.half_c = L.half_c;
        t.pos_f16 = L.pos_f16;
        t.sliding = L.sliding;
    }
    htab_.assign(sizeof(ModelTabH), 0);
    auto* mt = reinterpret_cast<ModelTabH*>(htab_.data());
    auto* s = static_cast<uint8_t*>(scratch_);
    void** ptrs[kSCount] = {&mt->xbuf, &mt->raw,  &mt->att,  &mt->o,   &mt->h1,  &mt->gu,  &mt->lg,  &mt->egu,
                            &mt->m,    &mt->e,    &mt->part, &mt->sync, &mt->tick, &mt->err, &mt->qs};
    for (int i = 0; i < kSCount; ++i)
        *ptrs[i] = s + off_[i];
    mt->w_final = w_.model[kModelFinalNorm];
    mt->lm = w_.model[kModelLmW];
    mt->lm_s = w_.model[kModelLmScale];
    mt->lm_z = w_.model[kModelLmZp];
    mt->cap_div = cap_div_;
    mt->cap_mul = cap_mul_;
    mt->attn_scale = scale_;
    mt->window = window_;
    mt->causal_off = causal_off_;
    mt->p_mul = p_mul_;
    mt->p_add = p_add_;
    mt->pos_stride = pos_stride_;
    mt->am_stride = am_stride_;
    mt->kv_cap = kv_cap_;
    mt->prof = prof_;
    mt->kmask = kmask_;
    MK_OK(usm_.memcpy(q_, CL_TRUE, ltab_, hltab_.data(), hltab_.size(), 0, nullptr, nullptr), "upload layer table");
    MK_OK(usm_.memcpy(q_, CL_TRUE, tab_, htab_.data(), htab_.size(), 0, nullptr, nullptr), "upload model table");
    return CL_SUCCESS;
}

int Gemma4ModelRuntime::EnsureCapacity(int L) {
    if (L <= kv_cap_)
        return CL_SUCCESS;
    const int cap = static_cast<int>(round_up(std::max(L, kv_cap_ * 2), 1024));
    for (auto& ly : layers_) {
        const size_t row = size_t(ly.D) * 2;
        for (void** buf : {&ly.kc, &ly.vc}) {
            void* nb = Alloc(size_t(ly.HK) * cap * row);
            if (!nb)
                return CL_OUT_OF_RESOURCES;
            if (*buf) {
                for (int h = 0; h < ly.HK && kv_valid_ > 0; ++h)
                    MK_OK(usm_.memcpy(q_, CL_FALSE, static_cast<uint8_t*>(nb) + size_t(h) * cap * row,
                                      static_cast<uint8_t*>(*buf) + size_t(h) * kv_cap_ * row, kv_valid_ * row, 0,
                                      nullptr, nullptr),
                          "kv copy");
                usm_.blocking_free(ctx_, *buf);
            }
            *buf = nb;
        }
    }
    kv_cap_ = cap;
    if (kmask_)
        usm_.blocking_free(ctx_, kmask_);
    kmask_ = Alloc(size_t(cap) * 4);
    if (!kmask_)
        return CL_OUT_OF_RESOURCES;
    return UploadTables();
}

int Gemma4ModelRuntime::Execute(const Gemma4RuntimeParams& p) {
    const int T = p.tokens, L = p.ka;
    if (p.batch != 1 || T <= 0 || L < T) {
        std::fprintf(stderr, "[Gemma4Model] unsupported call: batch %d, tokens %d, attention_mask %d\n", p.batch, T,
                     L);
        return CL_INVALID_VALUE;
    }
    const int past = L - T;
    if (past > kv_valid_) {
        std::fprintf(stderr, "[Gemma4Model] past length %d exceeds the cached %d tokens\n", past, kv_valid_);
        return CL_INVALID_VALUE;
    }
    const int rc = EnsureCapacity(L);
    if (rc != CL_SUCCESS)
        return rc;
    // OV_MEGAKERNEL_NO_DECODE=1 runs single tokens through the prompt path
    // (debugging aid: the per-layer kernels instead of mk_decode).
    static const bool no_decode = env_int("OV_MEGAKERNEL_NO_DECODE", 0) != 0;
    if (T == 1 && debug_cmp_ && !prof_on_) {
        // Residual entering every layer, prompt path vs mk_decode.
        std::vector<uint16_t> ref(size_t(nl_) * kHidden + size_t(nl_) * 8192),
            got(size_t(nl_) * kHidden + size_t(nl_) * 8192);
        dbg_x_ = &ref;
        int rc0 = Prompt(p, past, L);
        dbg_x_ = nullptr;
        if (rc0 == CL_SUCCESS)
            rc0 = Decode(p, past, L);
        clFinish(q_);
        usm_.memcpy(q_, CL_TRUE, got.data(), prof_, got.size() * 2, 0, nullptr, nullptr);
        auto h2f = [](uint16_t h) {
            const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
            float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? INFINITY : std::ldexp(float(m | 1024), int(e) - 25);
            return s ? -v : v;
        };
        std::fprintf(stderr, "[Gemma4Model] past %d: per-layer residual cos (prompt path vs mk_decode):", past);
        for (int l = 0; l < nl_; ++l) {
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i < kHidden; ++i) {
                const double a = h2f(ref[size_t(l) * kHidden + i]), b = h2f(got[size_t(l) * kHidden + i]);
                ab += a * b;
                aa += a * a;
                bb += b * b;
            }
            std::fprintf(stderr, " %d:%.4f", l, ab / std::sqrt(aa * bb + 1e-30));
        }
        std::fprintf(stderr, "\n  attention output cos:");
        for (int l = 0; l < nl_; ++l) {
            const int ka = layers_[l].HQ * layers_[l].D;
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i < ka; ++i) {
                const size_t o = size_t(nl_) * kHidden + size_t(l) * 8192 + i;
                const double a = h2f(ref[o]), b = h2f(got[o]);
                ab += a * b;
                aa += a * a;
                bb += b * b;
            }
            std::fprintf(stderr, " %d:%.4f", l, ab / std::sqrt(aa * bb + 1e-30));
        }
        for (int l = 0; l < nl_; ++l) {
            if (layers_[l].sliding)
                continue;
            const int D = layers_[l].D;
            std::fprintf(stderr, "\n  layer %d per-head cos:", l);
            for (int h = 0; h < layers_[l].HQ; ++h) {
                double ab = 0, aa = 0, bb = 0;
                for (int i = 0; i < D; ++i) {
                    const size_t o = size_t(nl_) * kHidden + size_t(l) * 8192 + size_t(h) * D + i;
                    const double a = h2f(ref[o]), b = h2f(got[o]);
                    ab += a * b;
                    aa += a * a;
                    bb += b * b;
                }
                std::fprintf(stderr, " %d:%.4f", h, ab / std::sqrt(aa * bb + 1e-30));
            }
            break;
        }
        std::fprintf(stderr, "\n");
        if (rc0 == CL_SUCCESS)
            kv_valid_ = L;
        return rc0;
    }
    const int r = T == 1 && !no_decode ? Decode(p, past, L) : Prompt(p, past, L);
    if (r == CL_SUCCESS)
        kv_valid_ = L;
    return r;
}

int Gemma4ModelRuntime::Decode(const Gemma4RuntimeParams& p, int past, int L) {
    // Key splits per kv head: a work-group has 16 (sliding) or 2 (full) key
    // lanes; aim for OV_MEGAKERNEL_ATT_KPS keys per lane.
    static const int kps_s = env_int("OV_MEGAKERNEL_ATT_KPS_S", 2);
    static const int kps_f = env_int("OV_MEGAKERNEL_ATT_KPS_F", 4);
    auto splits = [&](int nk, int keys_per_unit, int hk, int wgs) {
        const int lim = hk ? wgs / hk : 1;
        return std::max(1, std::min(lim, (nk + keys_per_unit - 1) / keys_per_unit));
    };
    // Sliding layers use at most 3/5 of the work-groups: the idle ones pull
    // o_proj into L2 meanwhile, which beat more key splits on B60 (287 and
    // 3348 token contexts). Full layers have far more keys and use them all.
    const int ns_s =
        splits(std::min(static_cast<int>(window_), past + 1), 16 * kps_s, max_kv_heads_[0], nwg_ * 3 / 5);
    const int ns_f = splits(past + 1, 2 * kps_f, max_kv_heads_[1], nwg_);
    ++epoch_;
    cl_kernel k = kDecode_;
    MK_OK(set_embeds_arg(usm_, k, 2, p), "embeds");
    MK_OK(usm_.set_arg(k, 3, p.in2), "pos");
    MK_OK(usm_.set_arg(k, 4, p.in3), "am");
    MK_OK(usm_.set_arg(k, 5, p.out), "logits");
    MK_OK(clSetKernelArg(k, 6, sizeof(int), &past), "past");
    MK_OK(clSetKernelArg(k, 7, sizeof(int), &L), "L");
    MK_OK(clSetKernelArg(k, 8, sizeof(int), &epoch_), "epoch");
    MK_OK(clSetKernelArg(k, 9, sizeof(int), &ns_s), "ns_s");
    MK_OK(clSetKernelArg(k, 10, sizeof(int), &ns_f), "ns_f");
    // Attention unit counters grow by NS per launch (all layers of a type alike).
    MK_OK(clSetKernelArg(k, 11, sizeof(int), &att_base_[0]), "att_base_s");
    MK_OK(clSetKernelArg(k, 12, sizeof(int), &att_base_[1]), "att_base_f");
    att_base_[0] += ns_s > 1 ? ns_s : 0;
    att_base_[1] += ns_f > 1 ? ns_f : 0;
    // attn_unit_xmx for the full-attention layers once their units hold >= 128
    // keys (bit 1; bit 0 would select it for the sliding ones). Below that, and
    // for the sliding layers (G = 2 fills 2 of the 8 DPAS rows), the scalar
    // unit measured as fast or faster on B60.
    const int xmx = past + 1 >= 128 * ns_f ? 2 : 0;
    MK_OK(clSetKernelArg(k, 13, sizeof(int), &xmx), "xmx");
    const size_t gws = size_t(nwg_) * kLws, lws = kLws;
    MK_OK(clEnqueueNDRangeKernel(q_, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr), "mk_decode");
    if (prof_on_)
        Profile();
    return CL_SUCCESS;
}

// OV_MEGAKERNEL_PROF=1: per-phase critical path from the barrier timestamps
// (the timestamps themselves shift small effects); =3: kernel time only.
void Gemma4ModelRuntime::Profile() {
    const int nb = nl_ * kPhasesPerLayer + 2;
    std::vector<uint64_t> t(size_t(nb) * nwg_ * 5 + size_t(nl_) * 16);
    clFinish(q_);
    usm_.memcpy(q_, CL_TRUE, t.data(), prof_, t.size() * 8, 0, nullptr, nullptr);
    auto at = [&](int b, int w, int which) { return double(t[(size_t(b) * nwg_ + w) * 5 + which]); };
    const int ks = nb - 1;  // kernel start / end slot
    if (prof_total_) {
        double tot = 0;
        for (int w = 0; w < nwg_; ++w)
            tot = std::max(tot, at(ks, w, 1) - at(ks, 0, 0));
        prof_d_.push_back(tot / 2.4e6);
        if (prof_d_.size() == 50) {
            std::sort(prof_d_.begin(), prof_d_.end());
            std::fprintf(stderr, "[Gemma4Model] kernel time over 50 steps: median %.3f ms, min %.3f\n", prof_d_[25],
                         prof_d_[0]);
            prof_d_.clear();
        }
        return;
    }
    if (prof_acc_.empty())
        prof_acc_.assign(5 * (kPhasesPerLayer + 3), 0.0);
    // Phase p of a layer ends at barrier p; G ends at the kernel end.
    for (int b = 0; b <= nl_ * kPhasesPerLayer; ++b) {
        const bool last = b == nl_ * kPhasesPerLayer;
        int ph = last ? kPhasesPerLayer : b % kPhasesPerLayer;
        if (ph == 1 && !layers_[b / kPhasesPerLayer].sliding)
            ph = kPhasesPerLayer + 2;  // attention of the full-attention layers
        double start_min = 1e30, arrive_max = 0, work = 0, release_min = 1e30, release_max = 0, pro = 0;
        for (int w = 0; w < nwg_; ++w) {
            const double s = b == 0 ? at(ks, w, 0) : at(b - 1, w, 1);
            const double a = last ? at(ks, w, 1) : at(b, w, 0);
            start_min = std::min(start_min, s);
            arrive_max = std::max(arrive_max, a);
            work += a - s;
            const double ws = at(last ? ks - 1 : b, w, 2);
            pro += ws > s && ws < a ? ws - s : 0;
            if (!last) {
                release_min = std::min(release_min, at(b, w, 1));
                release_max = std::max(release_max, at(b, w, 1));
            }
        }
        double* acc = &prof_acc_[5 * ph];
        acc[0] += arrive_max - start_min;                       // phase span (first start .. last arrival)
        acc[1] += work / nwg_;                                  // mean work-group busy time
        acc[2] += last ? 0 : release_min - arrive_max;          // barrier latency
        acc[3] += last ? 0 : release_max - release_min;         // release skew
        acc[4] += pro / nwg_;                                   // mean prologue (before streaming)
    }
    double tot = 0;
    for (int w = 0; w < nwg_; ++w)
        tot = std::max(tot, at(ks, w, 1) - at(ks, 0, 0));
    prof_acc_[5 * (kPhasesPerLayer + 1)] += tot;
    // Phase D: dense part, router wait, MoE part (mean over work-groups).
    if (prof_d_.empty())
        prof_d_.assign(4, 0.0);
    for (int l = 0; l < nl_; ++l) {
        const int b = l * kPhasesPerLayer + 3;
        for (int w = 0; w < nwg_; ++w) {
            prof_d_[0] += at(b, w, 3) - at(b, w, 2);
            prof_d_[1] += at(b, w, 4) - at(b, w, 3);
            prof_d_[2] += at(b, w, 0) - at(b, w, 4);
        }
    }
    if (prof_dbg_.empty())
        prof_dbg_.assign(16, 0.0);
    for (int l = 0; l < nl_; ++l) {
        const uint64_t* d = &t[size_t(nb) * nwg_ * 5 + size_t(l) * 16];
        const int pts[5] = {0, 3, 4, 5, 6};
        for (int i = 1; i < 5; ++i)
            prof_dbg_[(layers_[l].sliding ? 0 : 8) + i] += double(d[pts[i]] - d[pts[i - 1]]);
    }
    if (++prof_n_ % 20)
        return;
    const char* names[] = {"A qkv", "B attn(s)", "C oproj", "D gate/up", "E down", "G lm_head", "",
                           "B attn(f)"};
    int nfull = 0;
    for (const auto& ly : layers_)
        nfull += !ly.sliding;
    const double n = prof_n_;
    std::fprintf(stderr, "[Gemma4Model] profile over %d steps (timestamp ticks, per layer for A..F):\n", prof_n_);
    for (int ph = 0; ph <= kPhasesPerLayer + 2; ++ph) {
        if (ph == kPhasesPerLayer + 1)
            continue;
        const double div = n * (ph == kPhasesPerLayer ? 1 : ph == 1 ? nl_ - nfull : ph > kPhasesPerLayer ? nfull : nl_);
        const double* a = &prof_acc_[5 * ph];
        std::fprintf(stderr, "  %-10s span %7.2f us  mean-busy %7.2f  prologue %6.2f  barrier %5.2f  skew %5.2f\n",
                     names[ph], a[0] / div / 2400, a[1] / div / 2400, a[4] / div / 2400, a[2] / div / 2400,
                     a[3] / div / 2400);
    }
    std::fprintf(stderr, "  unit 0 (sliding | full) us: table+q, keys, wg combine, (last unit) merge:");
    for (int i = 1; i < 5; ++i)
        std::fprintf(stderr, " %.2f|%.2f", prof_dbg_[i] / n / (nl_ - nfull) / 2400, prof_dbg_[8 + i] / n / nfull / 2400);
    std::fprintf(stderr, "\n");
    prof_dbg_.assign(16, 0.0);
    std::fprintf(stderr, "  D: dense/router %.2f us, routing wait %.2f, MoE gate/up %.2f (work-group mean)\n",
                 prof_d_[0] / n / nl_ / nwg_ / 2400, prof_d_[1] / n / nl_ / nwg_ / 2400,
                 prof_d_[2] / n / nl_ / nwg_ / 2400);
    prof_d_.assign(4, 0.0);
    std::fprintf(stderr, "  kernel total %.3f ms (2.4 GHz ticks)\n", prof_acc_[5 * (kPhasesPerLayer + 1)] / n / 2.4e6);
    prof_acc_.assign(prof_acc_.size(), 0.0);
    prof_n_ = 0;
}

int Gemma4ModelRuntime::Prompt(const Gemma4RuntimeParams& p, int past, int L) {
    const int T = p.tokens;
    int hm_max = 0, o_max = 0;
    for (const auto& ly : layers_) {
        hm_max = std::max(hm_max, (ly.HQ + 2 * ly.HK) * ly.D);
        o_max = std::max(o_max, ly.HQ * ly.D);
    }
    const size_t need[kPfBufs] = {size_t(T) * kHidden * 2, size_t(T) * kHidden * 2, size_t(T) * hm_max * 2,
                                  size_t(T) * o_max * 2,   size_t(T) * L * 2,       size_t(T) * L * 2,
                                  size_t(T) * 4 * 4};
    for (int i = 0; i < kPfBufs; ++i)
        if (!Grow(pf_[i], pf_cap_[i], need[i]))
            return CL_OUT_OF_RESOURCES;
    void* xa = pf_[kXa];
    void* xb = pf_[kXb];
    int* rng = static_cast<int*>(pf_[kRange]);
    int* lo[2] = {rng, rng + 2 * T};
    int* hi[2] = {rng + T, rng + 3 * T};

    {
        const int n = T * kHidden;
        MK_OK(set_embeds_arg(usm_, kCvt_, 0, p), "cvt in");
        MK_OK(usm_.set_arg(kCvt_, 1, xa), "cvt out");
        MK_OK(clSetKernelArg(kCvt_, 2, sizeof(int), &n), "cvt n");
        const size_t gws = round_up(n, 256), lws = 256;
        MK_OK(clEnqueueNDRangeKernel(q_, kCvt_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr), "mk_cvt_embeds");
    }
    // Explicit [T, L] masks (vision groups included), and their visible key ranges.
    for (int i = 0; i < 2; ++i) {
        Gemma4RuntimeParams mp{};
        mp.in = p.in3;
        mp.in2 = p.in4;
        mp.out = pf_[i == 0 ? kMaskF : kMaskS];
        mp.tokens = 1;
        mp.ka = L;
        mp.s = T;
        const int rc = mask_rt_[i]->Execute(&mp);
        if (rc != CL_SUCCESS)
            return rc;
        MK_OK(usm_.set_arg(kMaskRange_, 0, mp.out), "range mask");
        MK_OK(clSetKernelArg(kMaskRange_, 1, sizeof(int), &T), "range T");
        MK_OK(clSetKernelArg(kMaskRange_, 2, sizeof(int), &L), "range L");
        MK_OK(usm_.set_arg(kMaskRange_, 3, lo[i]), "range lo");
        MK_OK(usm_.set_arg(kMaskRange_, 4, hi[i]), "range hi");
        const size_t gws[2] = {16, size_t(T)}, lws[2] = {16, 1};
        MK_OK(clEnqueueNDRangeKernel(q_, kMaskRange_, 2, nullptr, gws, lws, 0, nullptr, nullptr), "mk_mask_range");
    }

    for (int l = 0; l < nl_; ++l) {
        const Layer& ly = layers_[l];
        if (dbg_x_)
            usm_.memcpy(q_, CL_TRUE, dbg_x_->data() + size_t(l) * kHidden,
                        static_cast<uint8_t*>(xa) + size_t(T - 1) * kHidden * 2, kHidden * 2, 0, nullptr, nullptr);
        Gemma4RuntimeParams ap{};
        ap.in = xa;
        ap.in2 = p.in2;
        ap.out = pf_[kHm];
        ap.tokens = T;
        ap.s = T;
        int rc = attn_rt_[l]->Execute(&ap);
        if (rc != CL_SUCCESS)
            return rc;
        {
            cl_kernel k = kKvAppend_;
            MK_OK(usm_.set_arg(k, 0, pf_[kHm]), "kv hm");
            MK_OK(clSetKernelArg(k, 1, sizeof(int), &T), "kv T");
            MK_OK(clSetKernelArg(k, 2, sizeof(int), &ly.HQ), "kv HQ");
            MK_OK(clSetKernelArg(k, 3, sizeof(int), &ly.HK), "kv HK");
            MK_OK(clSetKernelArg(k, 4, sizeof(int), &ly.D), "kv D");
            MK_OK(usm_.set_arg(k, 5, ly.kc), "kv kc");
            MK_OK(usm_.set_arg(k, 6, ly.vc), "kv vc");
            MK_OK(clSetKernelArg(k, 7, sizeof(int), &kv_cap_), "kv cap");
            MK_OK(clSetKernelArg(k, 8, sizeof(int), &past), "kv past");
            const size_t gws[3] = {round_up(ly.D, 64), size_t(T), size_t(2 * ly.HK)}, lws[3] = {64, 1, 1};
            MK_OK(clEnqueueNDRangeKernel(q_, k, 3, nullptr, gws, lws, 0, nullptr, nullptr), "mk_kv_append");
        }
        {
            // XMX flash attention; OV_MEGAKERNEL_PF_ATTN_SIMPLE=1 keeps the scalar kernel.
            static const bool simple = std::getenv("OV_MEGAKERNEL_PF_ATTN_SIMPLE") != nullptr;
            cl_kernel k = simple ? kPfAttn_ : kPfFa_;
            const int s = ly.sliding ? 1 : 0;
            MK_OK(usm_.set_arg(k, 0, pf_[kHm]), "attn hm");
            MK_OK(usm_.set_arg(k, 1, ly.kc), "attn kc");
            MK_OK(usm_.set_arg(k, 2, ly.vc), "attn vc");
            MK_OK(usm_.set_arg(k, 3, pf_[s ? kMaskS : kMaskF]), "attn mask");
            MK_OK(usm_.set_arg(k, 4, lo[s]), "attn lo");
            MK_OK(usm_.set_arg(k, 5, hi[s]), "attn hi");
            MK_OK(clSetKernelArg(k, 6, sizeof(int), &T), "attn T");
            MK_OK(clSetKernelArg(k, 7, sizeof(int), &L), "attn L");
            MK_OK(clSetKernelArg(k, 8, sizeof(int), &ly.HQ), "attn HQ");
            MK_OK(clSetKernelArg(k, 9, sizeof(int), &ly.HK), "attn HK");
            MK_OK(clSetKernelArg(k, 10, sizeof(int), &ly.D), "attn D");
            MK_OK(clSetKernelArg(k, 11, sizeof(int), &kv_cap_), "attn cap");
            MK_OK(clSetKernelArg(k, 12, sizeof(float), &scale_), "attn scale");
            MK_OK(usm_.set_arg(k, 13, pf_[kO]), "attn O");
            const int qb = simple ? 512 / ly.D : 8;
            const int per_wg = simple ? 16 : 8;
            const size_t sgs = size_t(ly.HQ) * ((T + qb - 1) / qb) * (simple ? 1 : ly.D / 256);
            const size_t gws = round_up(sgs, per_wg) * 16, lws = per_wg * 16;
            MK_OK(clEnqueueNDRangeKernel(q_, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr), "mk_pf_attn");
        }
        if (dbg_x_)
            usm_.memcpy(q_, CL_TRUE, dbg_x_->data() + size_t(nl_) * kHidden + size_t(l) * 8192,
                        static_cast<uint8_t*>(pf_[kO]) + size_t(T - 1) * ly.HQ * ly.D * 2, size_t(ly.HQ) * ly.D * 2,
                        0, nullptr, nullptr);
        Gemma4RuntimeParams fp{};
        fp.in = pf_[kO];
        fp.in2 = xa;
        fp.out = xb;
        fp.tokens = T;
        fp.ka = ly.HQ * ly.D;
        rc = ffn_rt_[l]->Execute(&fp);
        if (rc != CL_SUCCESS)
            return rc;
        std::swap(xa, xb);
    }

    const int rows = last_only_ ? 1 : T;
    {
        cl_kernel k = kLmRows_;
        void* x = static_cast<uint8_t*>(xa) + size_t(T - rows) * kHidden * 2;
        MK_OK(usm_.set_arg(k, 0, tab_), "lm tab");
        MK_OK(usm_.set_arg(k, 1, x), "lm x");
        MK_OK(clSetKernelArg(k, 2, sizeof(int), &rows), "lm rows");
        MK_OK(usm_.set_arg(k, 3, p.out), "lm out");
        const size_t wgs = (size_t(vocab_) / 16 + 16 * kLmRb - 1) / (16 * kLmRb);
        const size_t gws[2] = {size_t((rows + kLmTb - 1) / kLmTb) * kLws, wgs}, lws[2] = {kLws, 1};
        MK_OK(clEnqueueNDRangeKernel(q_, k, 2, nullptr, gws, lws, 0, nullptr, nullptr), "mk_lm_head_rows");
    }
    return CL_SUCCESS;
}

void Gemma4ModelRuntime::Destroy() {
    attn_rt_.clear();
    ffn_rt_.clear();
    mask_rt_.clear();
    for (cl_kernel* k : {&kDecode_, &kCvt_, &kKvAppend_, &kMaskRange_, &kPfAttn_, &kPfFa_, &kLmRows_}) {
        if (*k)
            clReleaseKernel(*k);
        *k = nullptr;
    }
    if (prog_)
        clReleaseProgram(prog_);
    prog_ = nullptr;
    if (!ctx_ || !usm_.blocking_free)
        return;
    // In-flight kernels may still use these; the blocking free waits for them.
    for (auto& ly : layers_)
        for (void** b : {&ly.kc, &ly.vc})
            if (*b) {
                usm_.blocking_free(ctx_, *b);
                *b = nullptr;
            }
    for (void** b : {&scratch_, &tab_, &ltab_, &prof_, &kmask_})
        if (*b) {
            usm_.blocking_free(ctx_, *b);
            *b = nullptr;
        }
    for (auto& b : pf_)
        if (b) {
            usm_.blocking_free(ctx_, b);
            b = nullptr;
        }
}

}  // namespace mk
