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
// Compressed weights (GR_W_TYPE = 4-bit group 32 / 8-bit per-channel, see gr_common.cl): the lane loads GR_W*_KCH k values of its row as raw
// bytes per step and unpacks one K32 slice at a time into the same uint16 B operand; MMA and epilogues are shared.
// The per-subgroup (decode) DOWN / UP kernels fold the zero point and scale out of the B operand: u8 per channel into the
// epilogue (GR_WQ_FOLD), u4 per K32 group into the accumulator update (B = half(1024 + q), no per-weight arithmetic). The
// others (GR_Q_SHARE) dequantize each K32 slice once per work-group into SLM. RBAR is the same for every weight type.
//
// Common JIT: GR_BRANCHES, GR_HIDDEN, GR_W_TYPE (default f16). Dispatch: see the per-stage comments.

#include "gr_common.cl"

#define GR_CD (GR_BRANCHES * GR_HIDDEN)
#define GR_INV_C (1.0f / (float)GR_BRANCHES)
// Per-subgroup compressed kernels: the u8 ones also issue the activation tiles of the next weight chunk together with it; u4
// loads them per K32 slice, which is as fast.
#define GR_APF GR_WQ_FOLD

#if GR_PIPELINE_STAGE == 0
// ---------------------------------------------------------------------------------------------------------------
// RBAR. JIT: GR_EPS, GR_NORM_GAIN_PLUS_ONE, GR_CH (halves per chunk), GR_NV_CH (= GR_CH / 16), GR_NCH (= D / GR_CH),
// GR_RB_S (subgroups per branch row, divides GR_NCH), GR_GPF (gain chunks loaded before the norm
// reduction; bounded by the register budget).
// GR_RB_S subgroups per (branch, row): gws = [C * GR_RB_S * 16, rows, 1], lws = [C * GR_RB_S * 16, 1, 1].
// GR_KPERM: Rbar is stored K-permuted for the 4-bit decode GEMMs (see GR_KP).
// ---------------------------------------------------------------------------------------------------------------
#if GR_KPERM
#if GR_NV_CH != 8
    #error "gr_read_pipeline.cl: the K-permuted RBAR needs 128-half chunks"
#endif
// lane l owns the halves 8 l .. 8 l + 7 of a chunk (not 16 v + l), so the K permutation (GR_KP) is a register swizzle
#define GR_RB_LDU(P) as_ushort8(vload4(get_sub_group_local_id(), (const __global uint*)(P)))
#define GR_RB_ST(P, O) vstore4(as_uint4(GR_F2U(8, O).s04152637), get_sub_group_local_id(), (__global uint*)(P))
#else
#define GR_RB_LDU(P) GR_BRU(GR_NV_CH, P)
#define GR_RB_ST(P, O) GR_STF(GR_NV_CH, P, O)
#endif
#define GR_RB_LDF(P) GR_U2F(GR_NV_CH, GR_RB_LDU(P))
REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_rbar)(const __global ushort* residual, const __global ushort* norm_weight, __global ushort* rbar_out, uint num_rows) {
    const uint sg = get_sub_group_id();
    const uint c = sg / GR_RB_S;
    const uint row = get_group_id(1);
    if (c >= GR_BRANCHES || row >= num_rows) {
        return;
    }
#define GR_NCHP (GR_NCH / GR_RB_S)  // chunks of one subgroup
    const uint d_begin = (sg % GR_RB_S) * GR_NCHP * GR_CH;
    const size_t base = (size_t)row * GR_CD + c * GR_HIDDEN + d_begin;
    const uint gbase = c * GR_HIDDEN + d_begin;

#if GR_HIDDEN <= 4096
    // The subgroup's part of the branch row stays in registers: one read, normalized in place.
    GR_UT(GR_NV_CH) r[GR_NCHP];
    float sq = 0.0f;
#pragma unroll
    for (uint i = 0; i < GR_NCHP; i++) {
        r[i] = GR_RB_LDU(residual + base + i * GR_CH);
        const GR_FT(GR_NV_CH) f = GR_U2F(GR_NV_CH, r[i]);
        const GR_FT(GR_NV_CH) q = f * f;
        GR_ACC(GR_NV_CH, sq, q);
    }
    // The gain does not depend on the norm: the first GR_GPF chunks are loaded before the reduction to hide its latency.
#if GR_GPF > 0
    GR_UT(GR_NV_CH) gr_[GR_GPF];
#pragma unroll
    for (uint i = 0; i < GR_GPF; i++) {
        gr_[i] = GR_RB_LDU(norm_weight + gbase + i * GR_CH);
    }
#endif
#if GR_RB_S > 1
    __local float sq_part[GR_BRANCHES * GR_RB_S];
    sq = sub_group_reduce_add(sq);
    if (get_sub_group_local_id() == 0) {
        sq_part[sg] = sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sq = 0.0f;
    unroll_for (uint p = 0; p < GR_RB_S; p++) {
        sq += sq_part[c * GR_RB_S + p];
    }
#else
    sq = sub_group_reduce_add(sq);
#endif
    const float inv_rms = native_rsqrt(sq * (1.0f / (float)GR_HIDDEN) + GR_EPS);
#pragma unroll
    for (uint i = 0; i < GR_NCHP; i++) {
#if GR_GPF > 0
        GR_FT(GR_NV_CH) gain = i < GR_GPF ? GR_U2F(GR_NV_CH, gr_[i]) : GR_RB_LDF(norm_weight + gbase + i * GR_CH);
#else
        GR_FT(GR_NV_CH) gain = GR_RB_LDF(norm_weight + gbase + i * GR_CH);
#endif
#if GR_NORM_GAIN_PLUS_ONE
        gain = gain + 1.0f;
#endif
        const GR_FT(GR_NV_CH) f = GR_U2F(GR_NV_CH, r[i]);
        const GR_FT(GR_NV_CH) o = (f * inv_rms) * gain;
        GR_RB_ST(rbar_out + base + i * GR_CH, o);
    }
#else
    // Wide branches: two passes over the subgroup's part of the row, the second one re-reads it from cache.
    float sq = 0.0f;
    for (uint d = 0; d < GR_NCHP * GR_CH; d += GR_CH) {
        const GR_FT(GR_NV_CH) f = GR_RB_LDF(residual + base + d);
        const GR_FT(GR_NV_CH) q = f * f;
        GR_ACC(GR_NV_CH, sq, q);
    }
#if GR_RB_S > 1
    __local float sq_part[GR_BRANCHES * GR_RB_S];
    sq = sub_group_reduce_add(sq);
    if (get_sub_group_local_id() == 0) {
        sq_part[sg] = sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sq = 0.0f;
    unroll_for (uint p = 0; p < GR_RB_S; p++) {
        sq += sq_part[c * GR_RB_S + p];
    }
#else
    sq = sub_group_reduce_add(sq);
#endif
    const float inv_rms = native_rsqrt(sq * (1.0f / (float)GR_HIDDEN) + GR_EPS);
    for (uint d = 0; d < GR_NCHP * GR_CH; d += GR_CH) {
        const GR_FT(GR_NV_CH) f = GR_RB_LDF(residual + base + d);
        GR_FT(GR_NV_CH) gain = GR_RB_LDF(norm_weight + gbase + d);
#if GR_NORM_GAIN_PLUS_ONE
        gain = gain + 1.0f;
#endif
        const GR_FT(GR_NV_CH) o = (f * inv_rms) * gain;
        GR_RB_ST(rbar_out + base + d, o);
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
#if GR_WD_PERCH
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
#if GR_WD_PERCH
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv[j_], scv[j_]);
#else
                const half2 scv = GR_WQ_SCALE2(w_scale, GR_WQ_G(k >> 5, GR_WD_GS32), GR_LOWRANK, n0 + 16 * j_);
                const half2 zpv = GR_WQ_ZP2(w_zp, GR_WQ_G(k >> 5, GR_WD_GS32), GR_LOWRANK, n0 + 16 * j_, lane);
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv, scv);
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
#if GR_W_TYPE != GR_WT_F16 && (GR_KPT % GR_WD_KCH != 0)
    #error "gr_read_pipeline.cl: the per-subgroup K range must be a multiple of the weight load width"
#endif

#define GR_DN_LDA(AT, KK) GR_RD2D(GR_DOWN_TM, rbar, GR_CD * 2, num_tokens, GR_CD * 2, (KK), t0, AT)

#if GR_W_TYPE != GR_WT_F16 && !GR_WQ_FOLD
// Group of K32 slice s: W_inject has its own group size (GR_WI_GS32) in the write-gate tile
#if GR_PRODUCE_WRITE_GATE
#define GR_DN_GRP(s) (is_wgate ? GR_WQ_G(s, GR_WI_GS32) : GR_WQ_G(s, GR_WD_GS32))
#else
#define GR_DN_GRP(s) GR_WQ_G(s, GR_WD_GS32)
#endif
// scale / zp of the groups of the chunk at KK (one entry per K32 slice); W_inject's [G, C] rows are too short for block reads
#define GR_DN_LDSZ(SC, ZP, KK)                                                                                  \
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {                                                                \
        unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {                                                   \
            const uint i_ = GR_DN_GRP(((KK) >> 5) + s_) * qn + nrow[j_];                                        \
            SC[j_][s_] = qs[i_];                                                                                \
            ZP[j_][s_] = GR_HAS_ZP ? qz[GR_WQ_BITS == 4 ? (i_ >> 1) : i_] : 0;                                  \
        }                                                                                                       \
    }
#endif

#if GR_W_TYPE == GR_WT_F16
#define GR_DN_LDB(dst, j, KK) dst = vload16(0, (const __global uint*)(brow[j] + (KK)))

#define GR_DN_LOAD(AT, B, KK) { GR_DN_LDA(AT, KK); unroll_for (uint j_ = 0; j_ < GR_NT; j_++) { GR_DN_LDB(B[j_], j_, KK); } }
#endif

#define GR_DN_MMA_TO(ACC, AT, B)                                                                                \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(m_, (AT)));                                                     \
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {                                                        \
                (ACC)[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].lo), (ACC)[j_ * GR_MT + m_]);              \
            }                                                                                                   \
        }                                                                                                       \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(GR_MT + m_, (AT)));                                             \
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {                                                        \
                (ACC)[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].hi), (ACC)[j_ * GR_MT + m_]);              \
            }                                                                                                   \
        }                                                                                                       \
    }
#define GR_DN_MMA(AT, B) GR_DN_MMA_TO(acc, AT, B)

#if GR_W_TYPE != GR_WT_F16
// token row sums of the A tile: RS[m] = A[m-th 8 tokens][K32 slice] . 1
#define GR_DN_RS_TO(RS, AT)                                                                                     \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            (RS)[m_] = GR_MAD(as_short8(vload8(m_, (AT))), wq_ones, (RS)[m_]);                                  \
            (RS)[m_] = GR_MAD(as_short8(vload8(GR_MT + m_, (AT))), wq_ones, (RS)[m_]);                          \
        }                                                                                                       \
    }
#endif
#if GR_WQ_FOLD
#define GR_DN_RS(AT) GR_DN_RS_TO(acc + GR_NT * GR_MT, AT)
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_down_proj_dpas)(const __global ushort* rbar,
#if !GR_GATE_ONLY
#if GR_W_TYPE == GR_WT_F16
                          const __global ushort* w_down,
#else
                          const __global uchar* w_down,
                          const __global ushort* w_scale,
                          const __global uchar* w_zp,
#endif
#endif
#if GR_PRODUCE_WRITE_GATE
#if GR_W_TYPE == GR_WT_F16
                          const __global ushort* w_inject,
#else
                          const __global uchar* w_inject,
                          const __global ushort* s_inject,
                          const __global uchar* z_inject,
#endif
#endif
#if !GR_GATE_ONLY
                          __global ushort* t_out,
#endif
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
#if GR_GATE_ONLY
    // only the write-gate column tile: S = 2 * sigmoid(Rbar . W_inject^T / C)
    if (jt != 0) {
        return;
    }
    const bool is_wgate = true;
    const uint n0 = 0;
#elif GR_PRODUCE_WRITE_GATE
    if (jt > GR_LOWRANK / GR_DOWN_TN) {
        return;
    }
    const bool is_wgate = jt == GR_LOWRANK / GR_DOWN_TN;
    const uint n0 = is_wgate ? 0 : jt * GR_DOWN_TN;
#else
    if (jt >= GR_LOWRANK / GR_DOWN_TN) {
        return;
    }
    const bool is_wgate = false;
    const uint n0 = jt * GR_DOWN_TN;
#endif

    // B: lane n owns row (n0 + 16 * j + n) of W; the write-gate tile has only C rows, the rest repeat the last one.
    uint nrow[GR_NT];
#pragma unroll
    for (uint j = 0; j < GR_NT; j++) {
        nrow[j] = is_wgate ? min(lane, (uint)(GR_BRANCHES - 1)) : n0 + 16 * j + lane;
    }
#if GR_W_TYPE == GR_WT_F16
#if GR_GATE_ONLY
    const __global ushort* w = w_inject;
#elif GR_PRODUCE_WRITE_GATE
    const __global ushort* w = is_wgate ? w_inject : w_down;
#else
    const __global ushort* w = w_down;
#endif
    const __global ushort* brow[GR_NT];
#pragma unroll
    for (uint j = 0; j < GR_NT; j++) {
        brow[j] = w + (size_t)nrow[j] * GR_CD;
    }
#else
    // W_down has [G, L] scale / zp, W_inject [G, C]: the row stride qn of the scale / zp table differs
#if GR_GATE_ONLY
    const __global uchar* qw = w_inject;
    const __global ushort* qs = s_inject;
    const __global uchar* qz = z_inject;
    const uint qn = GR_BRANCHES;
#elif GR_PRODUCE_WRITE_GATE
    const __global uchar* qw = is_wgate ? w_inject : w_down;
    const __global ushort* qs = is_wgate ? s_inject : w_scale;
    const __global uchar* qz = is_wgate ? z_inject : w_zp;
    const uint qn = is_wgate ? GR_BRANCHES : GR_LOWRANK;
#else
    const __global uchar* qw = w_down;
    const __global ushort* qs = w_scale;
    const __global uchar* qz = w_zp;
    const uint qn = GR_LOWRANK;
#endif
    const __global uchar* brow[GR_NT];
#pragma unroll
    for (uint j = 0; j < GR_NT; j++) {
        brow[j] = qw + (size_t)nrow[j] * GR_WD_RB;
    }
#endif

    float8 acc[GR_NACC];
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        acc[i] = (float8)(0.0f);
    }
#if GR_W_TYPE != GR_WT_F16
    // a per-lane value (not a uniform constant), so the 8 registers of the row-sum operand stay live instead of being rebuilt per slice
    const int8 wq_ones = as_int8((uint8)(GR_WQ_ONES_U | ((num_tokens + lane) >> 31)));
#endif

    const uint k_begin = kid * GR_KPT;
    const uint k_end = k_begin + GR_KPT;
#if GR_W_TYPE == GR_WT_F16
    ushort at0[2 * GR_DOWN_TM];
#endif
    uint16 b0[GR_NT];
#if GR_W_TYPE != GR_WT_F16
    // the first weight chunk and its activation tiles are in flight before the loop
    uint rw[GR_NT][GR_WD_NRAW];
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
        GR_WQ_LOAD_RAW(rw[j_], GR_WD_NRAW, brow[j_], k_begin);
    }
#if GR_APF
    ushort atc[GR_WD_KCH / 32][2 * GR_DOWN_TM];
    unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {
        GR_DN_LDA(atc[s_], k_begin + 32 * s_);
    }
#endif
#if GR_W_TYPE != GR_WT_F16 && !GR_WQ_FOLD
    ushort scr[GR_NT][GR_WD_KCH / 32];
    uchar zpr[GR_NT][GR_WD_KCH / 32];
    GR_DN_LDSZ(scr, zpr, k_begin);
    uint zsh[GR_NT];
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
        zsh[j_] = GR_WQ_BITS == 4 ? (nrow[j_] & 1) * 4 : 0;  // qn is even, so the nibble of (g * qn + row) only depends on the row
    }
    // A . (bias + q) and the A row sums of the current group; scale / zp are applied once per group
#if GR_PRODUCE_WRITE_GATE
    const uint gs32 = is_wgate ? GR_WI_GS32 : GR_WD_GS32;
#else
    const uint gs32 = GR_WD_GS32;
#endif
    float8 tq[GR_NT * GR_MT];
    float8 rs[GR_MT];
    unroll_for (uint i_ = 0; i_ < GR_NT * GR_MT; i_++) {
        tq[i_] = (float8)(0.0f);
    }
    unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
        rs[m_] = (float8)(0.0f);
    }
#elif GR_W_TYPE != GR_WT_F16
    float sf[GR_NT];
    float zf[GR_NT];
    unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
        sf[j_] = convert_float(as_half(qs[nrow[j_]]));
        zf[j_] = convert_float(GR_WQ_ZPAT(qz, nrow[j_]));
    }
#endif
#endif

#if GR_W_TYPE == GR_WT_F16
    for (uint k = k_begin; k < k_end; k += 32) {
        GR_DN_LOAD(at0, b0, k);
        GR_DN_MMA(at0, b0);
    }
#else
    // The raw bytes (u4 scale / zp) and activation tiles of the next chunk are in flight while the current one is unpacked
    // and consumed (the last step re-reads the last chunk); a single chunk covering the K range is all in flight before the loop.
    for (uint k = k_begin; k < k_end; k += GR_WD_KCH) {
#if GR_WD_KCH < GR_KPT
        uint rwn[GR_NT][GR_WD_NRAW];
        const uint kn = min(k + GR_WD_KCH, k_end - GR_WD_KCH);
        unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
            GR_WQ_LOAD_RAW(rwn[j_], GR_WD_NRAW, brow[j_], kn);
        }
#if GR_W_TYPE != GR_WT_F16 && !GR_WQ_FOLD
        ushort scn[GR_NT][GR_WD_KCH / 32];
        uchar zpn[GR_NT][GR_WD_KCH / 32];
        GR_DN_LDSZ(scn, zpn, kn);
#endif
#if GR_APF
        ushort atn[GR_WD_KCH / 32][2 * GR_DOWN_TM];
        unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {
            GR_DN_LDA(atn[s_], kn + 32 * s_);
        }
#endif
#endif
        unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {
#if GR_APF
            const ushort* at0 = atc[s_];
#else
            ushort at0[2 * GR_DOWN_TM];
            GR_DN_LDA(at0, k + 32 * s_);
#endif
            unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
#if GR_WQ_FOLD
                b0[j_] = GR_WQ_U8B32(rw[j_], s_ * GR_WQ_RAW_PER_K32);
#elif GR_KPERM
                b0[j_] = GR_WQ_U4P32(rw[j_], s_ * GR_WQ_RAW_PER_K32);
#else
                b0[j_] = GR_WQ_GB32(rw[j_], s_ * GR_WQ_RAW_PER_K32);
#endif
            }
#if GR_WQ_FOLD
            GR_DN_MMA(at0, b0);
            GR_DN_RS(at0);
#else
            // acc += scale * (A . (bias + q) - (bias + zp) * rowsum(A)) per quantization group, no per-weight arithmetic
            GR_DN_MMA_TO(tq, at0, b0);
            GR_DN_RS_TO(rs, at0);
            const uint sl_ = (k >> 5) + s_;
            if ((sl_ + 1) % gs32 == 0 || sl_ + 1 == (k_end >> 5)) {
                unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
                    const float sc = convert_float(as_half(scr[j_][s_]));
                    const float zz = GR_WQ_BIAS + (float)GR_WQ_ZPLD(zpr[j_][s_], zsh[j_]);
                    unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
                        acc[j_ * GR_MT + m_] = fma((float8)(sc), fma((float8)(-zz), rs[m_], tq[j_ * GR_MT + m_]), acc[j_ * GR_MT + m_]);
                        tq[j_ * GR_MT + m_] = (float8)(0.0f);
                    }
                }
                unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
                    rs[m_] = (float8)(0.0f);
                }
            }
#endif
        }
#if GR_WD_KCH < GR_KPT
#if GR_APF
        unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {
            unroll_for (uint i_ = 0; i_ < 2 * GR_DOWN_TM; i_++) {
                atc[s_][i_] = atn[s_][i_];
            }
        }
#endif
        unroll_for (uint j_ = 0; j_ < GR_NT; j_++) {
            unroll_for (uint i_ = 0; i_ < GR_WD_NRAW; i_++) {
                rw[j_][i_] = rwn[j_][i_];
            }
#if GR_W_TYPE != GR_WT_F16 && !GR_WQ_FOLD
            unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {
                scr[j_][s_] = scn[j_][s_];
                zpr[j_][s_] = zpn[j_][s_];
            }
#endif
        }
#endif
    }
#endif

#if GR_DOWN_KSPLIT > 1
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        FUNC_CALL(gr_st8)(slm + (kid * GR_NACC + i) * 128, acc[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // subgroup i sums accumulator i over the K split (in K order, as one subgroup would) into the slot of split 0
    for (uint i = kid; i < GR_NACC; i += GR_DOWN_KSPLIT) {
        float8 s = FUNC_CALL(gr_ld8)(slm + i * 128);
        unroll_for (uint p = 1; p < GR_DOWN_KSPLIT; p++) {
            s += FUNC_CALL(gr_ld8)(slm + (p * GR_NACC + i) * 128);
        }
        FUNC_CALL(gr_st8)(slm + i * 128, s);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kid != 0) {
        return;
    }
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        acc[i] = FUNC_CALL(gr_ld8)(slm + i * 128);
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
#if !GR_GATE_ONLY
#pragma unroll
    for (uint m = 0; m < GR_MT; m++) {
#pragma unroll
        for (uint j = 0; j < GR_NT; j++) {
            const float8 x = acc[j * GR_MT + m] * inv_c;
#if GR_KPERM
            // T is stored K-permuted for the 4-bit decode UP_PROJ
            const float8 xs = intel_sub_group_shuffle(x, GR_KP(lane));
            const float8 t = xs * GR_SIGMOID(xs);
#else
            const float8 t = x * GR_SIGMOID(x);
#endif
            const ushort8 tu = as_ushort8(convert_half8_rte(t));
            ushort tmp[8];
            vstore8(tu, 0, tmp);
            // T is an internal buffer (64 B aligned): rows past num_tokens are dropped by the 2D store.
            intel_sub_group_2d_block_write_16b_8r16x1c((__global void*)t_out, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, (int2)(n0 + 16 * j, t0 + m * 8), tmp);
        }
    }
#endif
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
#if GR_WU_PERCH
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
#if GR_WU_PERCH
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv[c_], scv[c_]);
#else
                const half2 scv = GR_WQ_SCALE2(w_scale, GR_WQ_G(sl, GR_WU_GS32), GR_CD, c_ * GR_HIDDEN + d0);
                const half2 zpv = GR_WQ_ZP2(w_zp, GR_WQ_G(sl, GR_WU_GS32), GR_CD, c_ * GR_HIDDEN + d0, lane);
                const uint16 b = GR_WQ_DEQ32(rw, 0, zpv, scv);
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
#if GR_UP_BSPLIT
// ---------------------------------------------------------------------------------------------------------------
// UP_PROJ of the 4-bit decode GEMMs (GR_KPERM): one subgroup per branch of a 16-channel tile walks the whole low rank (lane =
// W_up row); the C read-gated branch partials of the mix meet in SLM. JIT: GR_LOWRANK, GR_UP_TM, GR_WU_*.
// gws = [(D / 16) * C * 16, ceil(tokens / TM), 1], lws = [C * 16, 1, 1]
// ---------------------------------------------------------------------------------------------------------------
#define GR_MT (GR_UP_TM / 8)
#if GR_W_TYPE != GR_WT_U4 || !GR_KPERM || GR_LOWRANK % GR_WU_KCH != 0
    #error "gr_read_pipeline.cl: the branch-split UP_PROJ is the 4-bit decode kernel"
#endif
// scale / (bias + zp) of the groups of the chunk at KK (one entry per K32 slice) for the subgroup's branch
#define GR_UPB_LDSZ(SC, ZZ, KK)                                                                                 \
    unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {                                                       \
        const uint zb_ = GR_WQ_G(((KK) >> 5) + s_, GR_WU_GS32) * GR_CD + cb * GR_HIDDEN + d0;                    \
        SC[s_] = convert_float(as_half(intel_sub_group_block_read_us(w_scale + zb_)));                          \
        ZZ[s_] = GR_WQ_BIAS + (float)GR_WQ_ZPAT(w_zp, zb_ + lane);                                              \
    }

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_up_proj_dpas)(const __global ushort* t_in,
                        const __global uchar* w_up,
                        const __global ushort* w_scale,
                        const __global uchar* w_zp,
                        const __global ushort* rbar,
                        __global ushort* mixed,
                        uint num_tokens) {
    __local float slm[GR_BRANCHES * GR_MT * 128] __attribute__((aligned(64)));
    const uint lane = get_sub_group_local_id();
    const uint cb = get_sub_group_id();
    const uint d0 = get_group_id(0) * 16;
    const uint t0 = get_group_id(1) * GR_UP_TM;
    if (d0 >= GR_HIDDEN || t0 >= num_tokens) {
        return;
    }
    const __global uchar* brow = w_up + (size_t)(cb * GR_HIDDEN + d0 + lane) * GR_WU_RB;
    const int8 wq_ones = as_int8((uint8)(GR_WQ_ONES_U | ((num_tokens + lane) >> 31)));
    // A . (bias + q) and the A row sums of the current group; scale / zp are applied once per group
    float8 acc[GR_MT];
    float8 tq[GR_MT];
    float8 rs[GR_MT];
    unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
        acc[m_] = (float8)(0.0f);
        tq[m_] = (float8)(0.0f);
        rs[m_] = (float8)(0.0f);
    }

    // the raw bytes and scale / zp of the next chunk are in flight while the current one is consumed
    uint rw[GR_WU_NRAW];
    GR_WQ_LOAD_RAW(rw, GR_WU_NRAW, brow, 0);
    float scf[GR_WU_KCH / 32];
    float zzf[GR_WU_KCH / 32];
    GR_UPB_LDSZ(scf, zzf, 0);
    for (uint k = 0; k < GR_LOWRANK; k += GR_WU_KCH) {
#if GR_WU_KCH < GR_LOWRANK
        uint rwn[GR_WU_NRAW];
        const uint kn = min(k + GR_WU_KCH, (uint)(GR_LOWRANK - GR_WU_KCH));
        GR_WQ_LOAD_RAW(rwn, GR_WU_NRAW, brow, kn);
        float scn[GR_WU_KCH / 32];
        float zzn[GR_WU_KCH / 32];
        GR_UPB_LDSZ(scn, zzn, kn);
#endif
        unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
            ushort at0[2 * GR_UP_TM];
            GR_RD2D(GR_UP_TM, t_in, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, k + 32 * s_, t0, at0);
            const uint16 b = GR_WQ_U4P32(rw, s_ * GR_WQ_RAW_PER_K32);
            unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
                const short8 alo = as_short8(vload8(m_, at0));
                const short8 ahi = as_short8(vload8(GR_MT + m_, at0));
                tq[m_] = GR_MAD(alo, as_int8(b.lo), tq[m_]);
                tq[m_] = GR_MAD(ahi, as_int8(b.hi), tq[m_]);
                rs[m_] = GR_MAD(alo, wq_ones, rs[m_]);
                rs[m_] = GR_MAD(ahi, wq_ones, rs[m_]);
            }
            const uint sl_ = (k >> 5) + s_;
            if ((sl_ + 1) % GR_WU_GS32 == 0 || sl_ + 1 == GR_LOWRANK / 32) {
                unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
                    acc[m_] = fma((float8)(scf[s_]), fma((float8)(-zzf[s_]), rs[m_], tq[m_]), acc[m_]);
                    tq[m_] = (float8)(0.0f);
                    rs[m_] = (float8)(0.0f);
                }
            }
        }
#if GR_WU_KCH < GR_LOWRANK
        unroll_for (uint i_ = 0; i_ < GR_WU_NRAW; i_++) {
            rw[i_] = rwn[i_];
        }
        unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
            scf[s_] = scn[s_];
            zzf[s_] = zzn[s_];
        }
#endif
    }

    // branch partial of the mix: sigmoid(read gate) * Rbar, Rbar stored K-permuted (channel d0 + lane at GR_KQ(lane))
    unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
        ushort rt[8];
        GR_RD2D1(8, rbar, GR_CD * 2, num_tokens, GR_CD * 2, cb * GR_HIDDEN + d0, t0 + m_ * 8, rt);
        const float8 rv = intel_sub_group_shuffle(convert_float8(as_half8(vload8(0, rt))), GR_KQ(lane));
        FUNC_CALL(gr_st8)(slm + (cb * GR_MT + m_) * 128, GR_SIGMOID(acc[m_]) * rv);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (cb != 0) {
        return;
    }
    unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
        float8 mix = FUNC_CALL(gr_ld8)(slm + m_ * 128);
        unroll_for (uint c_ = 1; c_ < GR_BRANCHES; c_++) {
            mix += FUNC_CALL(gr_ld8)(slm + (c_ * GR_MT + m_) * 128);
        }
        const ushort8 ou = as_ushort8(convert_half8_rte(mix * GR_INV_C));
        ushort tmp[8];
        vstore8(ou, 0, tmp);
        unroll_for (uint i_ = 0; i_ < 8; i_++) {
            const uint tok = t0 + m_ * 8 + i_;
            if (tok < num_tokens) {
                intel_sub_group_block_write_us(mixed + tok * GR_HIDDEN + d0, tmp[i_]);
            }
        }
    }
}

#else  // GR_UP_BSPLIT == 0
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

#define GR_UP_MMA_TO(ACC, AT, B)                                                                                \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(m_, (AT)));                                                     \
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                  \
                (ACC)[c_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[c_].lo), (ACC)[c_ * GR_MT + m_]);              \
            }                                                                                                   \
        }                                                                                                       \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            const short8 av_ = as_short8(vload8(GR_MT + m_, (AT)));                                             \
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                  \
                (ACC)[c_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[c_].hi), (ACC)[c_ * GR_MT + m_]);              \
            }                                                                                                   \
        }                                                                                                       \
    }
#define GR_UP_MMA(AT, B) GR_UP_MMA_TO(acc, AT, B)

#if GR_W_TYPE != GR_WT_F16
#define GR_UP_RS_TO(RS, AT)                                                                                     \
    {                                                                                                           \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                            \
            (RS)[m_] = GR_MAD(as_short8(vload8(m_, (AT))), wq_ones, (RS)[m_]);                                  \
            (RS)[m_] = GR_MAD(as_short8(vload8(GR_MT + m_, (AT))), wq_ones, (RS)[m_]);                          \
        }                                                                                                       \
    }
#endif
#if GR_WQ_FOLD
#define GR_UP_RS(AT) GR_UP_RS_TO(acc + GR_BRANCHES * GR_MT, AT)
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
#if GR_W_TYPE != GR_WT_F16
    const int8 wq_ones = as_int8((uint8)(GR_WQ_ONES_U | ((num_tokens + lane) >> 31)));
#endif

    // residual tiles of the mix (8 tokens per dpas tile, one 2D block read each); only the reducing subgroup needs them
    ushort rt[GR_BRANCHES][GR_MT][8];
#define GR_UP_LDRT()                                                                                            \
    if (kid == 0) {                                                                                             \
        unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                      \
            unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                        \
                GR_RD2D1(8, rbar, GR_CD * 2, num_tokens, GR_CD * 2, c_ * GR_HIDDEN + d0, t0 + m_ * 8, rt[c_][m_]); \
            }                                                                                                   \
        }                                                                                                       \
    }

    const uint k_begin = kid * GR_KPT;
    const uint k_end = k_begin + GR_KPT;
    uint16 b0[GR_BRANCHES];
#if GR_W_TYPE == GR_WT_F16
    ushort at0[2 * GR_UP_TM];
    for (uint k = k_begin; k < k_end; k += 32) {
        GR_UP_LOAD(at0, b0, k);
        GR_UP_MMA(at0, b0);
    }
#else
#define GR_UP_LDA(AT, KK) GR_RD2D(GR_UP_TM, t_in, GR_LOWRANK * 2, num_tokens, GR_LOWRANK * 2, (KK), t0, AT)
#if !GR_WQ_FOLD
// scale / (bias + zp) of the groups of the chunk at KK (one entry per K32 slice)
#define GR_UP_LDSZ(SC, ZZ, KK)                                                                                  \
    unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {                                                       \
        unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {                                                      \
            const uint zb_ = GR_WQ_G(((KK) >> 5) + s_, GR_WU_GS32) * GR_CD + c_ * GR_HIDDEN + d0;                \
            SC[c_][s_] = convert_float(as_half(intel_sub_group_block_read_us(w_scale + zb_)));                  \
            ZZ[c_][s_] = GR_WQ_BIAS + (float)GR_WQ_ZPAT(w_zp, zb_ + lane);                                      \
        }                                                                                                       \
    }
#endif
    // the first weight chunk, its activation tiles (and scale / zp) are in flight before the loop
    uint rw[GR_BRANCHES][GR_WU_NRAW];
    unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
        GR_WQ_LOAD_RAW(rw[c_], GR_WU_NRAW, brow[c_], k_begin);
    }
#if GR_APF
    ushort atc[GR_WU_KCH / 32][2 * GR_UP_TM];
    unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
        GR_UP_LDA(atc[s_], k_begin + 32 * s_);
    }
#endif
#if !GR_WQ_FOLD
    float scf[GR_BRANCHES][GR_WU_KCH / 32];
    float zzf[GR_BRANCHES][GR_WU_KCH / 32];
    GR_UP_LDSZ(scf, zzf, k_begin);
    // A . (bias + q) and the A row sums of the current group; scale / zp are applied once per group
    float8 tq[GR_BRANCHES * GR_MT];
    float8 rs[GR_MT];
    unroll_for (uint i_ = 0; i_ < GR_BRANCHES * GR_MT; i_++) {
        tq[i_] = (float8)(0.0f);
    }
    unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
        rs[m_] = (float8)(0.0f);
    }
#endif
#if GR_WQ_FOLD
    float sf[GR_BRANCHES];
    float zf[GR_BRANCHES];
    unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
        sf[c_] = GR_WQ_SCALEF(w_scale, c_ * GR_HIDDEN + d0);
        zf[c_] = GR_WQ_ZPF(w_zp, c_ * GR_HIDDEN + d0, lane);
    }
#endif
    for (uint k = k_begin; k < k_end; k += GR_WU_KCH) {
#if GR_WU_KCH < GR_KPT
        uint rwn[GR_BRANCHES][GR_WU_NRAW];
        const uint kn = min(k + GR_WU_KCH, k_end - GR_WU_KCH);
        unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
            GR_WQ_LOAD_RAW(rwn[c_], GR_WU_NRAW, brow[c_], kn);
        }
#if GR_APF
        ushort atn[GR_WU_KCH / 32][2 * GR_UP_TM];
        unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
            GR_UP_LDA(atn[s_], kn + 32 * s_);
        }
#endif
#if !GR_WQ_FOLD
        float scn[GR_BRANCHES][GR_WU_KCH / 32];
        float zzn[GR_BRANCHES][GR_WU_KCH / 32];
        GR_UP_LDSZ(scn, zzn, kn);
#endif
#endif
        unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
#if GR_APF
            const ushort* at0 = atc[s_];
#else
            ushort at0[2 * GR_UP_TM];
            GR_UP_LDA(at0, k + 32 * s_);
#endif
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
#if GR_WQ_FOLD
                b0[c_] = GR_WQ_U8B32(rw[c_], s_ * GR_WQ_RAW_PER_K32);
#else
                b0[c_] = GR_WQ_GB32(rw[c_], s_ * GR_WQ_RAW_PER_K32);
#endif
            }
#if GR_WQ_FOLD
            GR_UP_MMA(at0, b0);
            GR_UP_RS(at0);
#else
            // acc += scale * (A . (bias + q) - (bias + zp) * rowsum(A)) per quantization group, see DOWN_PROJ
            GR_UP_MMA_TO(tq, at0, b0);
            GR_UP_RS_TO(rs, at0);
            const uint sl_ = (k >> 5) + s_;
            if ((sl_ + 1) % GR_WU_GS32 == 0 || sl_ + 1 == (k_end >> 5)) {
                unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
                    unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
                        acc[c_ * GR_MT + m_] = fma((float8)(scf[c_][s_]), fma((float8)(-zzf[c_][s_]), rs[m_], tq[c_ * GR_MT + m_]), acc[c_ * GR_MT + m_]);
                        tq[c_ * GR_MT + m_] = (float8)(0.0f);
                    }
                }
                unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {
                    rs[m_] = (float8)(0.0f);
                }
            }
#endif
        }
#if GR_WU_KCH < GR_KPT
        unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
#if GR_APF
            unroll_for (uint i_ = 0; i_ < 2 * GR_UP_TM; i_++) {
                atc[s_][i_] = atn[s_][i_];
            }
#endif
#if !GR_WQ_FOLD
            unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
                scf[c_][s_] = scn[c_][s_];
                zzf[c_][s_] = zzn[c_][s_];
            }
#endif
        }
        unroll_for (uint c_ = 0; c_ < GR_BRANCHES; c_++) {
            unroll_for (uint i_ = 0; i_ < GR_WU_NRAW; i_++) {
                rw[c_][i_] = rwn[c_][i_];
            }
        }
#endif
    }
#endif
    GR_UP_LDRT();

#if GR_UP_KSPLIT > 1
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        FUNC_CALL(gr_st8)(slm + (kid * GR_NACC + i) * 128, acc[i]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // subgroup i sums accumulator i over the K split into the slot of split 0
    for (uint i = kid; i < GR_NACC; i += GR_UP_KSPLIT) {
        float8 s = FUNC_CALL(gr_ld8)(slm + i * 128);
        unroll_for (uint p = 1; p < GR_UP_KSPLIT; p++) {
            s += FUNC_CALL(gr_ld8)(slm + (p * GR_NACC + i) * 128);
        }
        FUNC_CALL(gr_st8)(slm + i * 128, s);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (kid != 0) {
        return;
    }
#pragma unroll
    for (uint i = 0; i < GR_NACC; i++) {
        acc[i] = FUNC_CALL(gr_ld8)(slm + i * 128);
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

#endif  // GR_UP_BSPLIT
#endif  // GR_Q_SHARE

#else
#    error "gr_read_pipeline.cl: GR_PIPELINE_STAGE must be 0, 1 or 2"
#endif
