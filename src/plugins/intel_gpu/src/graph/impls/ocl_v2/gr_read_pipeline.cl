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
// Compressed weights (GR_W_TYPE = u4 group 32 / u8 per-channel, see gr_common.cl): the lane loads GR_W*_KCH k values of its row as raw
// bytes per step and unpacks / dequantizes one K32 slice at a time into the same uint16 B operand; MMA and epilogues are shared.
// The per-subgroup (decode) DOWN / UP kernels fold the u8 zero point and scale into the epilogue (GR_WQ_FOLD), the others
// (GR_Q_SHARE) dequantize each K32 slice once per work-group into SLM. RBAR is the same for every weight type.
//
// Common JIT: GR_BRANCHES, GR_HIDDEN, GR_W_TYPE (default f16). Dispatch: see the per-stage comments.

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
#if GR_Q_SHARE
// ---------------------------------------------------------------------------------------------------------------
// DOWN_PROJ with a work-group shared dequantization. One work-group = GR_Q_ST token subgroups x GR_DOWN_KSPLIT K subgroups on
// one 32-column tile. Every iteration each subgroup dequantizes ONE K32 slice of its K range for both column tiles into SLM
// (f16, already in the lane = n B-operand layout), then all GR_Q_ST token subgroups run their dpas on all slices: the
// dequantization is paid once per work-group instead of once per token tile.
// JIT: GR_LOWRANK, GR_DOWN_TM (32), GR_DOWN_TN (32), GR_DOWN_KSPLIT, GR_Q_ST.
// gws = [NX * GR_Q_ST * KSPLIT * 16, ceil(ceil(tokens / TM) / GR_Q_ST), 1], lws = [GR_Q_ST * KSPLIT * 16, 1, 1], NX = L / TN.
// ---------------------------------------------------------------------------------------------------------------
#define GR_MT (GR_DOWN_TM / 8)
#define GR_NT (GR_DOWN_TN / 16)
#define GR_KPT (GR_CD / GR_DOWN_KSPLIT)
#define GR_NSL (GR_KPT / 32)
#define GR_ITERS ((GR_NSL + GR_Q_ST - 1) / GR_Q_ST)
#define GR_KREG (GR_Q_ST * GR_NT * 256)
#define GR_BUF_U (GR_DOWN_KSPLIT * GR_KREG)
#define GR_RED_U (GR_Q_ST * GR_DOWN_KSPLIT * GR_NT * GR_MT * 128)
#define GR_SLM_U (2 * GR_BUF_U > GR_RED_U ? 2 * GR_BUF_U : GR_RED_U)

#if GR_CD % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_KPT % 32 != 0
    #error "gr_read_pipeline.cl: C*D, L and the per-subgroup K range must be multiples of 32"
#endif
#if GR_PRODUCE_WRITE_GATE
    #error "gr_read_pipeline.cl: compressed weights do not support the write gate"
#endif

#define GR_DNS_MMA(AT, BLO, BHI)                                                                                \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(m_, (AT)));                                                     \
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {                                                        \
                acc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((BLO)[j_]), acc[j_ * GR_MT + m_]);                   \
            }                                                                                                   \
        }                                                                                                       \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(GR_MT + m_, (AT)));                                             \
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {                                                        \
                acc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((BHI)[j_]), acc[j_ * GR_MT + m_]);                   \
            }                                                                                                   \
        }                                                                                                       \
    }

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_down_proj_dpas)(const __global ushort* rbar,
                          const __global uchar* w_down,
                          const __global ushort* w_scale,
                          const __global uchar* w_zp,
                          __global ushort* t_out,
                          uint num_tokens) {
    __local uint slm[GR_SLM_U] __attribute__((aligned(64)));
    const uint lane = get_sub_group_local_id();
    const uint sg = get_sub_group_id();
    const uint tk = sg % GR_Q_ST;
    const uint kid = sg / GR_Q_ST;
    const uint jt = get_group_id(0);
    if (jt >= GR_LOWRANK / GR_DOWN_TN) {
        return;
    }
    const uint t0 = (get_group_id(1) * GR_Q_ST + tk) * GR_DOWN_TM;
    const uint n0 = jt * GR_DOWN_TN;

    const __global uchar* brow[GR_NT];
#pragma unroll
    for (uint j = 0; j < GR_NT; j++) {
        brow[j] = w_down + (size_t)(n0 + 16 * j + lane) * GR_WD_RB;
    }
    float8 acc[GR_NT * GR_MT];
#pragma unroll
    for (uint i = 0; i < GR_NT * GR_MT; i++) {
        acc[i] = (float8)(0.0f);
    }
#if GR_W_TYPE == GR_WT_U8
    half2 scv[GR_NT];
    half2 zpv[GR_NT];
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
        scv[j_] = GR_WQ_SCALE2(w_scale, 0, GR_LOWRANK, n0 + 16 * j_);
        zpv[j_] = GR_WQ_ZP2(w_zp, 0, GR_LOWRANK, n0 + 16 * j_, lane);
    }
#endif

    const uint k_begin = kid * GR_KPT;
    ushort at0[2 * GR_DOWN_TM];
    for (uint it = 0; it < GR_ITERS; it++) {
        __local uint* sb = slm + (it & 1) * GR_BUF_U + kid * GR_KREG;
        const uint sl = it * GR_Q_ST + tk;
        if (sl < GR_NSL) {
            const uint k = k_begin + sl * 32;
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
                uint rw[GR_WQ_RAW_PER_K32];
                GR_WQ_LOAD_RAW(rw, GR_WQ_RAW_PER_K32, brow[j_], k);
#if GR_W_TYPE == GR_WT_U4
                const half2 scv = GR_WQ_SCALE2(w_scale, k >> 5, GR_LOWRANK, n0 + 16 * j_);
                const half2 zpv = GR_WQ_ZP2(w_zp, k >> 5, GR_LOWRANK, n0 + 16 * j_, lane);
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv, scv);
#else
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv[j_], scv[j_]);
#endif
                GR_BLOCK_WRITEN_SLM(uint, 8, sb + (tk * GR_NT + j_) * 256, 0, b.lo);
                GR_BLOCK_WRITEN_SLM(uint, 8, sb + (tk * GR_NT + j_) * 256 + 128, 0, b.hi);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint nsl = min((uint)GR_Q_ST, (uint)(GR_NSL - it * GR_Q_ST));
        unroll_for (uint s_ = 0; s_ < GR_Q_ST; s_++) {
            if (s_ < nsl) {
                GR_RD2D(GR_DOWN_TM, rbar, GR_CD * 2, num_tokens, GR_CD * 2, k_begin + (it * GR_Q_ST + s_) * 32, t0, at0);
                uint8 blo[GR_NT];
                uint8 bhi[GR_NT];
                unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
                    blo[j_] = BLOCK_READN_SLM(uint, 8, (const __local uint*)(sb + (s_ * GR_NT + j_) * 256), 0);
                    bhi[j_] = BLOCK_READN_SLM(uint, 8, (const __local uint*)(sb + (s_ * GR_NT + j_) * 256 + 128), 0);
                }
                GR_DNS_MMA(at0, blo, bhi);
            }
        }
    }

#if GR_DOWN_KSPLIT > 1
    barrier(CLK_LOCAL_MEM_FENCE);
    __local float* red = (__local float*)slm;
#pragma unroll
    for (uint i = 0; i < GR_NT * GR_MT; i++) {
        FUNC_CALL(gr_st8)(red + ((tk * GR_DOWN_KSPLIT + kid) * GR_NT * GR_MT + i) * 128, acc[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kid != 0) {
        return;
    }
    for (uint s = 1; s < GR_DOWN_KSPLIT; s++) {
#pragma unroll
        for (uint i = 0; i < GR_NT * GR_MT; i++) {
            acc[i] += FUNC_CALL(gr_ld8)(red + ((tk * GR_DOWN_KSPLIT + s) * GR_NT * GR_MT + i) * 128);
        }
    }
#endif

    const float inv_c = GR_INV_C;
#pragma unroll
    for (uint m = 0; m < GR_MT; m++) {
#pragma unroll
        for (uint j = 0; j < GR_NT; j++) {
            const float8 x = acc[j * GR_MT + m] * inv_c;
            const float8 t = x * GR_SIGMOID(x);
            const ushort8 tu = as_ushort8(convert_half8_rte(t));
            ushort tmp[8];
            vstore8(tu, 0, tmp);
            // rows past num_tokens are dropped by the 2D store
            intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)t_out, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, (int2)(n0 + 16 * j, t0 + m * 8), tmp);
        }
    }
}

#else  // GR_Q_SHARE == 0: every subgroup loads (and dequantizes) its own weights (few token tiles)
// ---------------------------------------------------------------------------------------------------------------
// DOWN_PROJ. JIT: GR_LOWRANK, GR_DOWN_TM (tokens per subgroup), GR_DOWN_TN (32), GR_DOWN_KSPLIT, GR_PRODUCE_WRITE_GATE.
// KSPLIT == 1: gws = [NX * 16, NY, 1], lws = [16, 1, 1];  KSPLIT > 1: gws = [NX * KSPLIT * 16, NY, 1], lws = [KSPLIT * 16, 1, 1]
// with NX = L / TN + write-gate tile, NY = ceil(tokens / TM). KSPLIT subgroups split the K reduction through SLM.
// ---------------------------------------------------------------------------------------------------------------
#define GR_MT (GR_DOWN_TM / 8)
#define GR_NT (GR_DOWN_TN / 16)
#define GR_KPT (GR_CD / GR_DOWN_KSPLIT)
#define GR_NACC ((GR_NT + GR_WQ_FOLD) * GR_MT)  // + the row sums of the folded u8 zero point

#if GR_CD % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_KPT % 32 != 0
    #error "gr_read_pipeline.cl: C*D, L and the per-subgroup K range must be multiples of 32"
#endif
#if GR_W_TYPE != GR_WT_F16 && GR_PRODUCE_WRITE_GATE
    #error "gr_read_pipeline.cl: compressed weights do not support the write gate"
#endif
#if GR_W_TYPE != GR_WT_F16 && (GR_KPT % GR_WD_KCH != 0)
    #error "gr_read_pipeline.cl: the per-subgroup K range must be a multiple of the weight load width"
#endif

#if GR_W_TYPE == GR_WT_F16
#define GR_DN_LDB(dst, j, KK) dst = vload16(0, (const __global uint*)(brow[j] + (KK)))

#define GR_DN_LOAD(AT, B, KK) { GR_RD2D(GR_DOWN_TM, rbar, GR_CD * 2, num_tokens, GR_CD * 2, (KK), t0, AT); unroll_for (uint j_ = 0; j_ < GR_NT; j_++) { GR_DN_LDB(B[j_], j_, KK); } }
#endif

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

#if GR_WQ_FOLD
#define GR_DN_RS(AT)                                                                                            \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            acc[GR_NT * GR_MT + m_] = GR_MAD(as_short8(vload8(m_, (AT))), GR_WQ_ONES, acc[GR_NT * GR_MT + m_]); \
            acc[GR_NT * GR_MT + m_] = GR_MAD(as_short8(vload8(GR_MT + m_, (AT))), GR_WQ_ONES, acc[GR_NT * GR_MT + m_]); \
        }                                                                                                       \
    }
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_down_proj_dpas)(const __global ushort* rbar,
#if GR_W_TYPE == GR_WT_F16
                          const __global ushort* w_down,
#else
                          const __global uchar* w_down,
                          const __global ushort* w_scale,
                          const __global uchar* w_zp,
#endif
#if GR_PRODUCE_WRITE_GATE
                          const __global ushort* w_inject,
#endif
                          __global ushort* t_out,
#if GR_PRODUCE_WRITE_GATE
                          __global ushort* write_gate_out,
#endif
                          uint num_tokens) {
#if GR_DOWN_KSPLIT > 1
    __local float slm[GR_DOWN_KSPLIT * GR_NACC * 128] __attribute__((aligned(64)));
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
#else
    if (jt >= GR_LOWRANK / GR_DOWN_TN) {
        return;
    }
#if GR_W_TYPE == GR_WT_F16
    const __global ushort* w = w_down;
#endif
    const uint n0 = jt * GR_DOWN_TN;
#endif

    // B: lane n owns row (n0 + 16 * j + n) of W; the write-gate tile has only C rows, the rest repeat the last one.
#if GR_W_TYPE == GR_WT_F16
    const __global ushort* brow[GR_NT];
#else
    const __global uchar* brow[GR_NT];
#endif
#pragma unroll
    for (uint j = 0; j < GR_NT; j++) {
#if GR_PRODUCE_WRITE_GATE
        const uint row = is_wgate ? min(lane, (uint)(GR_BRANCHES - 1)) : n0 + 16 * j + lane;
#else
        const uint row = n0 + 16 * j + lane;
#endif
#if GR_W_TYPE == GR_WT_F16
        brow[j] = w + (size_t)row * GR_CD;
#else
        brow[j] = w_down + (size_t)row * GR_WD_RB;
#endif
    }

    float8 acc[GR_NACC];
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        acc[i] = (float8)(0.0f);
    }

    const uint k_begin = kid * GR_KPT;
    const uint k_end = k_begin + GR_KPT;
    ushort at0[2 * GR_DOWN_TM];
    uint16 b0[GR_NT];
#if GR_W_TYPE == GR_WT_F16
    for (uint k = k_begin; k < k_end; k += 32) {
        GR_DN_LOAD(at0, b0, k);
        GR_DN_MMA(at0, b0);
    }
#else
#if GR_WQ_FOLD
    float sf[GR_NT];
    float zf[GR_NT];
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
        sf[j_] = GR_WQ_SCALEF(w_scale, n0 + 16 * j_);
        zf[j_] = GR_WQ_ZPF(w_zp, n0 + 16 * j_, lane);
    }
#elif GR_W_TYPE == GR_WT_U8
    // per-channel: one scale / zp per output channel for the whole K
    half2 scv[GR_NT];
    half2 zpv[GR_NT];
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
        scv[j_] = GR_WQ_SCALE2(w_scale, 0, GR_LOWRANK, n0 + 16 * j_);
        zpv[j_] = GR_WQ_ZP2(w_zp, 0, GR_LOWRANK, n0 + 16 * j_, lane);
    }
#endif
    // The raw bytes of the next chunk are in flight while the current one is unpacked and consumed (the last step re-reads
    // the last chunk).
    uint rw[GR_NT][GR_WD_NRAW];
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
        GR_WQ_LOAD_RAW(rw[j_], GR_WD_NRAW, brow[j_], k_begin);
    }
    for (uint k = k_begin; k < k_end; k += GR_WD_KCH) {
        uint rwn[GR_NT][GR_WD_NRAW];
        const uint kn = min(k + GR_WD_KCH, k_end - GR_WD_KCH);
        unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
            GR_WQ_LOAD_RAW(rwn[j_], GR_WD_NRAW, brow[j_], kn);
        }
        unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {
            GR_RD2D(GR_DOWN_TM, rbar, GR_CD * 2, num_tokens, GR_CD * 2, k + 32 * s_, t0, at0);
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
#if GR_WQ_FOLD
                b0[j_] = GR_WQ_U8B32(rw[j_], s_ * GR_WQ_RAW_PER_K32);
#elif GR_W_TYPE == GR_WT_U4
                const half2 scv = GR_WQ_SCALE2(w_scale, (k >> 5) + s_, GR_LOWRANK, n0 + 16 * j_);
                const half2 zpv = GR_WQ_ZP2(w_zp, (k >> 5) + s_, GR_LOWRANK, n0 + 16 * j_, lane);
                b0[j_] = GR_WQ_DEQ32(rw[j_], s_ * GR_WQ_RAW_PER_K32, zpv, scv);
#else
                b0[j_] = GR_WQ_DEQ32(rw[j_], s_ * GR_WQ_RAW_PER_K32, zpv[j_], scv[j_]);
#endif
            }
            GR_DN_MMA(at0, b0);
#if GR_WQ_FOLD
            GR_DN_RS(at0);
#endif
        }
        unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
            unroll_for (uint i_ = 0; i_ < GR_WD_NRAW; i_++) {
                rw[j_][i_] = rwn[j_][i_];
            }
        }
    }
#endif

#if GR_DOWN_KSPLIT > 1
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        FUNC_CALL(gr_st8)(slm + (kid * GR_NACC + i) * 128, acc[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kid != 0) {
        return;
    }
    for (uint s = 1; s < GR_DOWN_KSPLIT; s++) {
#pragma unroll
        for (uint i = 0; i < GR_NACC; i++) {
            acc[i] += FUNC_CALL(gr_ld8)(slm + (s * GR_NACC + i) * 128);
        }
    }
#endif
#if GR_WQ_FOLD
#pragma unroll
    for (uint m = 0; m < GR_MT; m++) {
#pragma unroll
        for (uint j = 0; j < GR_NT; j++) {
            acc[j * GR_MT + m] = (acc[j * GR_MT + m] - zf[j] * acc[GR_NT * GR_MT + m]) * sf[j];
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

#endif  // GR_Q_SHARE

#elif GR_PIPELINE_STAGE == 2
#if GR_Q_SHARE
// ---------------------------------------------------------------------------------------------------------------
// UP_PROJ with a work-group shared dequantization (see DOWN_PROJ): GR_Q_ST token subgroups on one 16-channel tile of all
// branches; every iteration each subgroup dequantizes one K32 slice of all GR_BRANCHES weight tiles into SLM. No split-K.
// JIT: GR_LOWRANK, GR_UP_TM, GR_UP_KSPLIT (1), GR_Q_ST.
// gws = [(D / 16) * GR_Q_ST * 16, ceil(ceil(tokens / TM) / GR_Q_ST), 1], lws = [GR_Q_ST * 16, 1, 1].
// ---------------------------------------------------------------------------------------------------------------
#define GR_MT (GR_UP_TM / 8)
#define GR_NSL (GR_LOWRANK / 32)
#define GR_ITERS ((GR_NSL + GR_Q_ST - 1) / GR_Q_ST)
#define GR_BUF_U (GR_Q_ST * GR_BRANCHES * 256)

#if GR_CD % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_HIDDEN % 16 != 0 || GR_UP_KSPLIT != 1
    #error "gr_read_pipeline.cl: C*D and L must be multiples of 32, D of 16, and the shared UP_PROJ has no split-K"
#endif

#define GR_UPS_MMA(AT, BLO, BHI)                                                                                \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(m_, (AT)));                                                     \
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                  \
                acc[c_ * GR_MT + m_] = GR_MAD(av_, as_int8((BLO)[c_]), acc[c_ * GR_MT + m_]);                   \
            }                                                                                                   \
        }                                                                                                       \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(GR_MT + m_, (AT)));                                             \
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                  \
                acc[c_ * GR_MT + m_] = GR_MAD(av_, as_int8((BHI)[c_]), acc[c_ * GR_MT + m_]);                   \
            }                                                                                                   \
        }                                                                                                       \
    }

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_up_proj_dpas)(const __global ushort* t_in,
                        const __global uchar* w_up,
                        const __global ushort* w_scale,
                        const __global uchar* w_zp,
                        const __global ushort* rbar,
                        __global ushort* mixed,
                        uint num_tokens) {
    __local uint slm[2 * GR_BUF_U] __attribute__((aligned(64)));
    const uint lane = get_sub_group_local_id();
    const uint tk = get_sub_group_id();
    const uint d0 = get_group_id(0) * 16;
    if (d0 >= GR_HIDDEN) {
        return;
    }
    const uint t0 = (get_group_id(1) * GR_Q_ST + tk) * GR_UP_TM;

    const __global uchar* brow[GR_BRANCHES];
#pragma unroll
    for (uint c = 0; c < GR_BRANCHES; c++) {
        brow[c] = w_up + (size_t)(c * GR_HIDDEN + d0 + lane) * GR_WU_RB;
    }
    float8 acc[GR_BRANCHES * GR_MT];
#pragma unroll
    for (uint i = 0; i < GR_BRANCHES * GR_MT; i++) {
        acc[i] = (float8)(0.0f);
    }
#if GR_W_TYPE == GR_WT_U8
    half2 scv[GR_BRANCHES];
    half2 zpv[GR_BRANCHES];
    unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
        scv[c_] = GR_WQ_SCALE2(w_scale, 0, GR_CD, c_ * GR_HIDDEN + d0);
        zpv[c_] = GR_WQ_ZP2(w_zp, 0, GR_CD, c_ * GR_HIDDEN + d0, lane);
    }
#endif

    ushort at0[2 * GR_UP_TM];
    for (uint it = 0; it < GR_ITERS; it++) {
        __local uint* sb = slm + (it & 1) * GR_BUF_U;
        const uint sl = it * GR_Q_ST + tk;
        if (sl < GR_NSL) {
            const uint k = sl * 32;
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
                uint rw[GR_WQ_RAW_PER_K32];
                GR_WQ_LOAD_RAW(rw, GR_WQ_RAW_PER_K32, brow[c_], k);
#if GR_W_TYPE == GR_WT_U4
                const half2 scv = GR_WQ_SCALE2(w_scale, sl, GR_CD, c_ * GR_HIDDEN + d0);
                const half2 zpv = GR_WQ_ZP2(w_zp, sl, GR_CD, c_ * GR_HIDDEN + d0, lane);
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv, scv);
#else
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv[c_], scv[c_]);
#endif
                GR_BLOCK_WRITEN_SLM(uint, 8, sb + (tk * GR_BRANCHES + c_) * 256, 0, b.lo);
                GR_BLOCK_WRITEN_SLM(uint, 8, sb + (tk * GR_BRANCHES + c_) * 256 + 128, 0, b.hi);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint nsl = min((uint)GR_Q_ST, (uint)(GR_NSL - it * GR_Q_ST));
        unroll_for (uint s_ = 0; s_ < GR_Q_ST; s_++) {
            if (s_ < nsl) {
                GR_RD2D(GR_UP_TM, t_in, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, (it * GR_Q_ST + s_) * 32, t0, at0);
                uint8 blo[GR_BRANCHES];
                uint8 bhi[GR_BRANCHES];
                unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
                    blo[c_] = BLOCK_READN_SLM(uint, 8, (const __local uint*)(sb + (s_ * GR_BRANCHES + c_) * 256), 0);
                    bhi[c_] = BLOCK_READN_SLM(uint, 8, (const __local uint*)(sb + (s_ * GR_BRANCHES + c_) * 256 + 128), 0);
                }
                GR_UPS_MMA(at0, blo, bhi);
            }
        }
    }

    ushort rt[GR_BRANCHES][GR_MT][8];
#pragma unroll
    for (uint c = 0; c < GR_BRANCHES; c++) {
#pragma unroll
        for (uint m = 0; m < GR_MT; m++) {
            GR_RD2D1(8, rbar, GR_CD * 2, num_tokens, GR_CD * 2, c * GR_HIDDEN + d0, t0 + m * 8, rt[c][m]);
        }
    }
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

#else  // GR_Q_SHARE == 0
// ---------------------------------------------------------------------------------------------------------------
// UP_PROJ. JIT: GR_LOWRANK, GR_UP_TM, GR_UP_KSPLIT. One subgroup owns TM tokens x 16 channels x all branches, so the
// read gate never leaves registers.
// KSPLIT > 1: gws = [(D / 16) * KSPLIT * 16, NY, 1], lws = [KSPLIT * 16, 1, 1]
// KSPLIT == 1: gws = [(D / 16) * 16, ceil(NY / lws_y) * lws_y, 1], lws = [16, lws_y, 1]
// ---------------------------------------------------------------------------------------------------------------
#define GR_MT (GR_UP_TM / 8)
#define GR_KPT (GR_LOWRANK / GR_UP_KSPLIT)
#define GR_NACC ((GR_BRANCHES + GR_WQ_FOLD) * GR_MT)  // + the row sums of the folded u8 zero point

#if GR_CD % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_KPT % 32 != 0 || GR_HIDDEN % 16 != 0
    #error "gr_read_pipeline.cl: C*D and L must be multiples of 32, D of 16"
#endif
#if GR_W_TYPE != GR_WT_F16 && (GR_KPT % GR_WU_KCH != 0)
    #error "gr_read_pipeline.cl: the per-subgroup K range must be a multiple of the weight load width"
#endif

#if GR_W_TYPE == GR_WT_F16
#define GR_UP_LDB(dst, c, KK) dst = vload16(0, (const __global uint*)(brow[c] + (KK)))

#define GR_UP_LOAD(AT, B, KK) { GR_RD2D(GR_UP_TM, t_in, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, (KK), t0, AT); unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) { GR_UP_LDB(B[c_], c_, KK); } }
#endif

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

#if GR_WQ_FOLD
#define GR_UP_RS(AT)                                                                                            \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            acc[GR_BRANCHES * GR_MT + m_] = GR_MAD(as_short8(vload8(m_, (AT))), GR_WQ_ONES, acc[GR_BRANCHES * GR_MT + m_]); \
            acc[GR_BRANCHES * GR_MT + m_] = GR_MAD(as_short8(vload8(GR_MT + m_, (AT))), GR_WQ_ONES, acc[GR_BRANCHES * GR_MT + m_]); \
        }                                                                                                       \
    }
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_up_proj_dpas)(const __global ushort* t_in,
#if GR_W_TYPE == GR_WT_F16
                        const __global ushort* w_up,
#else
                        const __global uchar* w_up,
                        const __global ushort* w_scale,
                        const __global uchar* w_zp,
#endif
                        const __global ushort* rbar,
                        __global ushort* mixed,
                        uint num_tokens) {
#if GR_UP_KSPLIT > 1
    __local float slm[GR_UP_KSPLIT * GR_NACC * 128] __attribute__((aligned(64)));
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

#if GR_W_TYPE == GR_WT_F16
    const __global ushort* brow[GR_BRANCHES];
#else
    const __global uchar* brow[GR_BRANCHES];
#endif
#pragma unroll
    for (uint c = 0; c < GR_BRANCHES; c++) {
#if GR_W_TYPE == GR_WT_F16
        brow[c] = w_up + (size_t)(c * GR_HIDDEN + d0 + lane) * GR_LOWRANK;
#else
        brow[c] = w_up + (size_t)(c * GR_HIDDEN + d0 + lane) * GR_WU_RB;
#endif
    }

    float8 acc[GR_NACC];
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        acc[i] = (float8)(0.0f);
    }

    const uint k_begin = kid * GR_KPT;
    const uint k_end = k_begin + GR_KPT;
    ushort at0[2 * GR_UP_TM];
    uint16 b0[GR_BRANCHES];
#if GR_W_TYPE == GR_WT_F16
    for (uint k = k_begin; k < k_end; k += 32) {
        GR_UP_LOAD(at0, b0, k);
        GR_UP_MMA(at0, b0);
    }
#else
#if GR_WQ_FOLD
    float sf[GR_BRANCHES];
    float zf[GR_BRANCHES];
    unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
        sf[c_] = GR_WQ_SCALEF(w_scale, c_ * GR_HIDDEN + d0);
        zf[c_] = GR_WQ_ZPF(w_zp, c_ * GR_HIDDEN + d0, lane);
    }
#elif GR_W_TYPE == GR_WT_U8
    half2 scv[GR_BRANCHES];
    half2 zpv[GR_BRANCHES];
    unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
        scv[c_] = GR_WQ_SCALE2(w_scale, 0, GR_CD, c_ * GR_HIDDEN + d0);
        zpv[c_] = GR_WQ_ZP2(w_zp, 0, GR_CD, c_ * GR_HIDDEN + d0, lane);
    }
#endif
    uint rw[GR_BRANCHES][GR_WU_NRAW];
    unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
        GR_WQ_LOAD_RAW(rw[c_], GR_WU_NRAW, brow[c_], k_begin);
    }
    for (uint k = k_begin; k < k_end; k += GR_WU_KCH) {
        uint rwn[GR_BRANCHES][GR_WU_NRAW];
        const uint kn = min(k + GR_WU_KCH, k_end - GR_WU_KCH);
        unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
            GR_WQ_LOAD_RAW(rwn[c_], GR_WU_NRAW, brow[c_], kn);
        }
        unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
            GR_RD2D(GR_UP_TM, t_in, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, k + 32 * s_, t0, at0);
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
#if GR_WQ_FOLD
                b0[c_] = GR_WQ_U8B32(rw[c_], s_ * GR_WQ_RAW_PER_K32);
#elif GR_W_TYPE == GR_WT_U4
                const half2 scv = GR_WQ_SCALE2(w_scale, (k >> 5) + s_, GR_CD, c_ * GR_HIDDEN + d0);
                const half2 zpv = GR_WQ_ZP2(w_zp, (k >> 5) + s_, GR_CD, c_ * GR_HIDDEN + d0, lane);
                b0[c_] = GR_WQ_DEQ32(rw[c_], s_ * GR_WQ_RAW_PER_K32, zpv, scv);
#else
                b0[c_] = GR_WQ_DEQ32(rw[c_], s_ * GR_WQ_RAW_PER_K32, zpv[c_], scv[c_]);
#endif
            }
            GR_UP_MMA(at0, b0);
#if GR_WQ_FOLD
            GR_UP_RS(at0);
#endif
        }
        unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
            unroll_for (uint i_ = 0; i_ < GR_WU_NRAW; i_++) {
                rw[c_][i_] = rwn[c_][i_];
            }
        }
    }
#endif

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
    for (uint i = 0; i < GR_NACC; i++) {
        FUNC_CALL(gr_st8)(slm + (kid * GR_NACC + i) * 128, acc[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kid != 0) {
        return;
    }
    for (uint s = 1; s < GR_UP_KSPLIT; s++) {
#pragma unroll
        for (uint i = 0; i < GR_NACC; i++) {
            acc[i] += FUNC_CALL(gr_ld8)(slm + (s * GR_NACC + i) * 128);
        }
    }
#endif
#if GR_WQ_FOLD
#pragma unroll
    for (uint c = 0; c < GR_BRANCHES; c++) {
#pragma unroll
        for (uint m = 0; m < GR_MT; m++) {
            acc[c * GR_MT + m] = (acc[c * GR_MT + m] - zf[c] * acc[GR_BRANCHES * GR_MT + m]) * sf[c];
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

#endif  // GR_Q_SHARE

#else
#    error "gr_read_pipeline.cl: GR_PIPELINE_STAGE must be 0, 1 or 2"
#endif
