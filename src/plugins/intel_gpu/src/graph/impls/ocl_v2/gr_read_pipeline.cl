// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GatedResidual READ / MIX as a systolic (dpas) pipeline of three dispatches sharing this template:
//   GR_PIPELINE_STAGE == 0  RBAR       branch-wise grouped RMSNorm -> Rbar scratch (global memory)
//   GR_PIPELINE_STAGE == 1  DOWN_PROJ  T = SiLU((Rbar . W_down^T) / C) (+ write gate S as one extra column tile)
//   GR_PIPELINE_STAGE == 2  UP_PROJ    mixed = (1/C) * sum_c sigmoid(T . W_up^T)[c] * Rbar[c]
//
// One subgroup is one hardware thread. dpas operands (SIMD16, M = 8 tokens, N = 16, K = 16):
//   A (short8)  a[i] = A[token i][k = lane]      -> a whole TM x K32 tile per 2D block read (rows past the end read 0)
//   B (int8)    b[j] = pack(B[2j][n = lane], B[2j+1][n = lane]); W rows are K-contiguous, so every lane reads its own
//               row of W with one 64 B vload16 (K32) - no transposed load needed
//   C (float8)  c[i] = C[token i][n = lane]
// The accumulator layout of one GEMM is the A layout of the next one. The loads of the next K32 step are issued before the
// dpas of the current one (two register buffers).
//
// Common JIT: GR_BRANCHES, GR_HIDDEN. Dispatch: see the per-stage comments.

#include "gr_common.cl"

#define GR_CD (GR_BRANCHES * GR_HIDDEN)
#define GR_INV_C (1.0f / (float)GR_BRANCHES)

#if GR_PIPELINE_STAGE == 0
// ---------------------------------------------------------------------------------------------------------------
// RBAR. JIT: GR_EPS, GR_NORM_GAIN_PLUS_ONE, GR_CH (halves per chunk), GR_NV_CH (= GR_CH / 16), GR_NCH (= D / GR_CH),
// GR_GPF (gain chunks loaded before the norm reduction; bounded by the register budget).
// One subgroup per (branch, row): gws = [C * 16, rows, 1], lws = [C * 16, 1, 1].
// ---------------------------------------------------------------------------------------------------------------
REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_rbar)(const __global ushort* residual, const __global ushort* norm_weight, __global ushort* rbar_out, uint num_rows) {
    const uint c = get_sub_group_id();
    const uint row = get_group_id(1);
    if (c >= GR_BRANCHES || row >= num_rows) {
        return;
    }
    const size_t base = (size_t)row * GR_CD + c * GR_HIDDEN;
    const uint gbase = c * GR_HIDDEN;

#if GR_HIDDEN <= 4096
    // The whole branch row stays in registers: one read, normalized in place.
    GR_UT(GR_NV_CH) r[GR_NCH];
    float sq = 0.0f;
#pragma unroll
    for (uint i = 0; i < GR_NCH; i++) {
        r[i] = GR_BRU(GR_NV_CH, residual + base + i * GR_CH);
        const GR_FT(GR_NV_CH) f = GR_U2F(GR_NV_CH, r[i]);
        const GR_FT(GR_NV_CH) q = f * f;
        GR_ACC(GR_NV_CH, sq, q);
    }
    // The gain does not depend on the norm: the first GR_GPF chunks are loaded before the reduction to hide its latency.
#if GR_GPF > 0
    GR_UT(GR_NV_CH) gr_[GR_GPF];
#pragma unroll
    for (uint i = 0; i < GR_GPF; i++) {
        gr_[i] = GR_BRU(GR_NV_CH, norm_weight + gbase + i * GR_CH);
    }
#endif
    const float inv_rms = native_rsqrt(sub_group_reduce_add(sq) * (1.0f / (float)GR_HIDDEN) + GR_EPS);
#pragma unroll
    for (uint i = 0; i < GR_NCH; i++) {
#if GR_GPF > 0
        GR_FT(GR_NV_CH) gain = i < GR_GPF ? GR_U2F(GR_NV_CH, gr_[i]) : GR_LDF(GR_NV_CH, norm_weight + gbase + i * GR_CH);
#else
        GR_FT(GR_NV_CH) gain = GR_LDF(GR_NV_CH, norm_weight + gbase + i * GR_CH);
#endif
#if GR_NORM_GAIN_PLUS_ONE
        gain = gain + 1.0f;
#endif
        const GR_FT(GR_NV_CH) f = GR_U2F(GR_NV_CH, r[i]);
        const GR_FT(GR_NV_CH) o = (f * inv_rms) * gain;
        GR_STF(GR_NV_CH, rbar_out + base + i * GR_CH, o);
    }
#else
    // Wide branches: two passes, the second one re-reads the row from cache.
    float sq = 0.0f;
    for (uint d = 0; d < GR_HIDDEN; d += GR_CH) {
        const GR_FT(GR_NV_CH) f = GR_LDF(GR_NV_CH, residual + base + d);
        const GR_FT(GR_NV_CH) q = f * f;
        GR_ACC(GR_NV_CH, sq, q);
    }
    const float inv_rms = native_rsqrt(sub_group_reduce_add(sq) * (1.0f / (float)GR_HIDDEN) + GR_EPS);
    for (uint d = 0; d < GR_HIDDEN; d += GR_CH) {
        const GR_FT(GR_NV_CH) f = GR_LDF(GR_NV_CH, residual + base + d);
        GR_FT(GR_NV_CH) gain = GR_LDF(GR_NV_CH, norm_weight + gbase + d);
#if GR_NORM_GAIN_PLUS_ONE
        gain = gain + 1.0f;
#endif
        const GR_FT(GR_NV_CH) o = (f * inv_rms) * gain;
        GR_STF(GR_NV_CH, rbar_out + base + d, o);
    }
#endif
}

#elif GR_PIPELINE_STAGE == 1
// ---------------------------------------------------------------------------------------------------------------
// DOWN_PROJ. JIT: GR_LOWRANK, GR_DOWN_TM (tokens per subgroup), GR_DOWN_TN (32), GR_DOWN_KSPLIT, GR_PRODUCE_WRITE_GATE.
// KSPLIT == 1: gws = [NX * 16, NY, 1], lws = [16, 1, 1];  KSPLIT > 1: gws = [NX * KSPLIT * 16, NY, 1], lws = [KSPLIT * 16, 1, 1]
// with NX = L / TN + write-gate tile, NY = ceil(tokens / TM). KSPLIT subgroups split the K reduction through SLM.
// ---------------------------------------------------------------------------------------------------------------
#define GR_MT (GR_DOWN_TM / 8)
#define GR_NT (GR_DOWN_TN / 16)
#define GR_KPT (GR_CD / GR_DOWN_KSPLIT)

#if GR_CD % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_KPT % 32 != 0
    #error "gr_read_pipeline.cl: C*D, L and the per-subgroup K range must be multiples of 32"
#endif

#define GR_DN_LDB(dst, j, KK) dst = vload16(0, (const __global uint*)(brow[j] + (KK)))

#define GR_DN_LOAD(AT, B, KK) { GR_RD2D(GR_DOWN_TM, rbar, GR_CD * 2, num_tokens, GR_CD * 2, (KK), t0, AT); unroll_for (uint j_ = 0; j_ < GR_NT; j_++) { GR_DN_LDB(B[j_], j_, KK); } }

#define GR_DN_MMA(AT, B)                                                                                        \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(m_, (AT)));                                                     \
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {                                                        \
                acc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].lo), acc[j_ * GR_MT + m_]);                  \
            }                                                                                                   \
        }                                                                                                       \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(GR_MT + m_, (AT)));                                             \
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {                                                        \
                acc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].hi), acc[j_ * GR_MT + m_]);                  \
            }                                                                                                   \
        }                                                                                                       \
    }

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_down_proj_dpas)(const __global ushort* rbar,
                          const __global ushort* w_down,
#if GR_PRODUCE_WRITE_GATE
                          const __global ushort* w_inject,
#endif
                          __global ushort* t_out,
#if GR_PRODUCE_WRITE_GATE
                          __global ushort* write_gate_out,
#endif
                          uint num_tokens) {
#if GR_DOWN_KSPLIT > 1
    __local float slm[GR_DOWN_KSPLIT * GR_MT * GR_NT * 128] __attribute__((aligned(64)));
#endif
    const uint lane = get_sub_group_local_id();
    const uint kid = get_sub_group_id();
    const uint jt = get_group_id(0);
    const uint t0 = get_group_id(1) * GR_DOWN_TM;
    if (t0 >= num_tokens) {
        return;
    }
#if GR_PRODUCE_WRITE_GATE
    if (jt > GR_LOWRANK / GR_DOWN_TN) {
        return;
    }
    const bool is_wgate = jt == GR_LOWRANK / GR_DOWN_TN;
    const __global ushort* w = is_wgate ? w_inject : w_down;
    const uint n0 = is_wgate ? 0 : jt * GR_DOWN_TN;
    const uint w_rows = is_wgate ? GR_BRANCHES : GR_LOWRANK;
#else
    if (jt >= GR_LOWRANK / GR_DOWN_TN) {
        return;
    }
    const __global ushort* w = w_down;
    const uint n0 = jt * GR_DOWN_TN;
    const uint w_rows = GR_LOWRANK;
#endif

    // B: lane n owns row (n0 + 16 * j + n) of W; the write-gate tile has only C rows, the rest repeat the last one.
    const __global ushort* brow[GR_NT];
#pragma unroll
    for (uint j = 0; j < GR_NT; j++) {
#if GR_PRODUCE_WRITE_GATE
        const uint row = is_wgate ? min(lane, (uint)(GR_BRANCHES - 1)) : n0 + 16 * j + lane;
#else
        const uint row = n0 + 16 * j + lane;
#endif
        brow[j] = w + (size_t)row * GR_CD;
    }

    float8 acc[GR_NT * GR_MT];
#pragma unroll
    for (uint i = 0; i < GR_NT * GR_MT; i++) {
        acc[i] = (float8)(0.0f);
    }

    const uint k_begin = kid * GR_KPT;
    const uint k_end = k_begin + GR_KPT;
    ushort at0[2 * GR_DOWN_TM];
    uint16 b0[GR_NT];
    for (uint k = k_begin; k < k_end; k += 32) {
        GR_DN_LOAD(at0, b0, k);
        GR_DN_MMA(at0, b0);
    }

#if GR_DOWN_KSPLIT > 1
#pragma unroll
    for (uint i = 0; i < GR_NT * GR_MT; i++) {
        FUNC_CALL(gr_st8)(slm + (kid * GR_NT * GR_MT + i) * 128, acc[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kid != 0) {
        return;
    }
    for (uint s = 1; s < GR_DOWN_KSPLIT; s++) {
#pragma unroll
        for (uint i = 0; i < GR_NT * GR_MT; i++) {
            acc[i] += FUNC_CALL(gr_ld8)(slm + (s * GR_NT * GR_MT + i) * 128);
        }
    }
#endif

    const float inv_c = GR_INV_C;
#if GR_PRODUCE_WRITE_GATE
    if (is_wgate) {
        if (lane < GR_BRANCHES) {
#pragma unroll
            for (uint m = 0; m < GR_MT; m++) {
                const float8 x = acc[m] * inv_c;
                const float8 s = 2.0f * GR_SIGMOID(x);
                float tmp[8];
                vstore8(s, 0, tmp);
#pragma unroll
                for (uint i = 0; i < 8; i++) {
                    const uint tok = t0 + m * 8 + i;
                    if (tok < num_tokens) {
                        write_gate_out[tok * GR_BRANCHES + lane] = as_ushort(convert_half_rte(tmp[i]));
                    }
                }
            }
        }
        return;
    }
#endif
#pragma unroll
    for (uint m = 0; m < GR_MT; m++) {
#pragma unroll
        for (uint j = 0; j < GR_NT; j++) {
            const float8 x = acc[j * GR_MT + m] * inv_c;
            const float8 t = x * GR_SIGMOID(x);
            const ushort8 tu = as_ushort8(convert_half8_rte(t));
            ushort tmp[8];
            vstore8(tu, 0, tmp);
            // T is an internal buffer (64 B aligned): rows past num_tokens are dropped by the 2D store.
            intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)t_out, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, (int2)(n0 + 16 * j, t0 + m * 8), tmp);
        }
    }
}

#elif GR_PIPELINE_STAGE == 2
// ---------------------------------------------------------------------------------------------------------------
// UP_PROJ. JIT: GR_LOWRANK, GR_UP_TM, GR_UP_KSPLIT. One subgroup owns TM tokens x 16 channels x all branches, so the
// read gate never leaves registers.
// KSPLIT > 1: gws = [(D / 16) * KSPLIT * 16, NY, 1], lws = [KSPLIT * 16, 1, 1]
// KSPLIT == 1: gws = [(D / 16) * 16, ceil(NY / lws_y) * lws_y, 1], lws = [16, lws_y, 1]
// ---------------------------------------------------------------------------------------------------------------
#define GR_MT (GR_UP_TM / 8)
#define GR_KPT (GR_LOWRANK / GR_UP_KSPLIT)

#if GR_CD % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_KPT % 32 != 0 || GR_HIDDEN % 16 != 0
    #error "gr_read_pipeline.cl: C*D and L must be multiples of 32, D of 16"
#endif

#define GR_UP_LDB(dst, c, KK) dst = vload16(0, (const __global uint*)(brow[c] + (KK)))

#define GR_UP_LOAD(AT, B, KK) { GR_RD2D(GR_UP_TM, t_in, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, (KK), t0, AT); unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) { GR_UP_LDB(B[c_], c_, KK); } }

#define GR_UP_MMA(AT, B)                                                                                        \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(m_, (AT)));                                                     \
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                  \
                acc[c_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[c_].lo), acc[c_ * GR_MT + m_]);                  \
            }                                                                                                   \
        }                                                                                                       \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(GR_MT + m_, (AT)));                                             \
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                  \
                acc[c_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[c_].hi), acc[c_ * GR_MT + m_]);                  \
            }                                                                                                   \
        }                                                                                                       \
    }

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_up_proj_dpas)(const __global ushort* t_in, const __global ushort* w_up, const __global ushort* rbar, __global ushort* mixed, uint num_tokens) {
#if GR_UP_KSPLIT > 1
    __local float slm[GR_UP_KSPLIT * GR_BRANCHES * GR_MT * 128] __attribute__((aligned(64)));
    const uint kid = get_sub_group_id();
#else
    const uint kid = 0;
#endif
    const uint lane = get_sub_group_local_id();
    const uint d0 = get_group_id(0) * 16;
    const uint t0 = get_global_id(1) * GR_UP_TM;
    if (d0 >= GR_HIDDEN || t0 >= num_tokens) {
        return;
    }

    const __global ushort* brow[GR_BRANCHES];
#pragma unroll
    for (uint c = 0; c < GR_BRANCHES; c++) {
        brow[c] = w_up + (size_t)(c * GR_HIDDEN + d0 + lane) * GR_LOWRANK;
    }

    float8 acc[GR_BRANCHES * GR_MT];
#pragma unroll
    for (uint i = 0; i < GR_BRANCHES * GR_MT; i++) {
        acc[i] = (float8)(0.0f);
    }

    const uint k_begin = kid * GR_KPT;
    const uint k_end = k_begin + GR_KPT;
    ushort at0[2 * GR_UP_TM];
    uint16 b0[GR_BRANCHES];
    for (uint k = k_begin; k < k_end; k += 32) {
        GR_UP_LOAD(at0, b0, k);
        GR_UP_MMA(at0, b0);
    }

    // residual tiles of the mix (8 tokens per dpas tile, one 2D block read each); only the reducing subgroup needs them
    ushort rt[GR_BRANCHES][GR_MT][8];
    if (kid == 0) {
#pragma unroll
        for (uint c = 0; c < GR_BRANCHES; c++) {
#pragma unroll
            for (uint m = 0; m < GR_MT; m++) {
                GR_RD2D1(8, rbar, GR_CD * 2, num_tokens, GR_CD * 2, c * GR_HIDDEN + d0, t0 + m * 8, rt[c][m]);
            }
        }
    }

#if GR_UP_KSPLIT > 1
#pragma unroll
    for (uint i = 0; i < GR_BRANCHES * GR_MT; i++) {
        FUNC_CALL(gr_st8)(slm + (kid * GR_BRANCHES * GR_MT + i) * 128, acc[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kid != 0) {
        return;
    }
    for (uint s = 1; s < GR_UP_KSPLIT; s++) {
#pragma unroll
        for (uint i = 0; i < GR_BRANCHES * GR_MT; i++) {
            acc[i] += FUNC_CALL(gr_ld8)(slm + (s * GR_BRANCHES * GR_MT + i) * 128);
        }
    }
#endif

    float8 mix[GR_MT];
#pragma unroll
    for (uint m = 0; m < GR_MT; m++) {
        mix[m] = (float8)(0.0f);
    }
#pragma unroll
    for (uint c = 0; c < GR_BRANCHES; c++) {
#pragma unroll
        for (uint m = 0; m < GR_MT; m++) {
            mix[m] += GR_SIGMOID(acc[c * GR_MT + m]) * convert_float8(as_half8(vload8(0, rt[c][m])));
        }
    }

#pragma unroll
    for (uint m = 0; m < GR_MT; m++) {
        const ushort8 ou = as_ushort8(convert_half8_rte(mix[m] * GR_INV_C));
        ushort tmp[8];
        vstore8(ou, 0, tmp);
#pragma unroll
        for (uint i = 0; i < 8; i++) {
            const uint tok = t0 + m * 8 + i;
            if (tok < num_tokens) {
                intel_sub_group_block_write_us(mixed + tok * GR_HIDDEN + d0, tmp[i]);
            }
        }
    }
}

#else
#    error "gr_read_pipeline.cl: GR_PIPELINE_STAGE must be 0, 1 or 2"
#endif
