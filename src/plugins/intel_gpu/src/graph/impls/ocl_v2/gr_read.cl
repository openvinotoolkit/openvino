// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GatedResidual READ / MIX, fused scalar fallback (pre-Xe2 or shapes the systolic path rejects).
//
//   Rbar = branch-wise RMSNorm(R) * gain  ->  t = SiLU(W_down Rbar / C)  ->  G = sigmoid(W_up t)
//   mixed = (1/C) * sum_c G_c * Rbar_c,   optionally S = 2 * sigmoid(W_inject Rbar / C)
//
// One work-group owns GR_READ_ROWS token rows; one subgroup is one hardware thread. Rbar and t live in SLM.
// JIT: GR_BRANCHES (4), GR_HIDDEN, GR_LOWRANK, GR_EPS, GR_PRODUCE_WRITE_GATE,
//      GR_WG_SIZE (subgroups per work-group), GR_READ_ROWS, GR_NV_D / GR_NV_C / GR_NV_L (= chunk halves / 16 of the
//      three K loops over D, C*D and L).
// Compressed weights (GR_W_TYPE != 0, see gr_common.cl: i4 / u4 / i8 / u8 rows [N, K] with f16 scale / zero point [G, N]) are
// dequantized per element in registers with the arithmetic of the decompression graph, half(half(q - zp) * scale), so the results
// are bit-identical to this kernel fed the decompressed f16 weights. JIT: GR_W_SIGNED, GR_HAS_ZP, GR_RD_GSD / GR_RD_GSU /
// GR_RD_GSI (group size along K of W_down / W_up / W_inject).
// Dispatch: gws = [WG * 16, ceil(rows / RB), 1], lws = [WG * 16, 1, 1].

#include "gr_common.cl"

#if GR_BRANCHES < 1 || GR_BRANCHES > 4
    #error "gr_read.cl: branch count must be in [1, 4]"
#endif
#if GR_PRODUCE_WRITE_GATE && GR_BRANCHES != 4
    #error "gr_read.cl: packed write-gate store expects 4 branches"
#endif
#if GR_HIDDEN % 16 != 0 || GR_LOWRANK % 16 != 0
    #error "gr_read.cl: hidden size and low rank must be multiples of 16"
#endif

#define GR_CD (GR_BRANCHES * GR_HIDDEN)
#define GR_NOUT (GR_LOWRANK + (GR_PRODUCE_WRITE_GATE ? GR_BRANCHES : 0))
#define GR_TSTRIDE ((GR_NOUT + 3) / 4 * 4)
#define GR_NJ 4
#define GR_KD (GR_NV_D * 16)
#define GR_KC (GR_NV_C * 16)
#define GR_KL (GR_NV_L * 16)
#define GR_NK (GR_LOWRANK / 16)
#define GR_INV_C (1.0f / (float)GR_BRANCHES)

#if GR_W_TYPE == GR_WT_F16
    #define GR_WPTR const __global ushort*
#else
    #define GR_WPTR const __global uchar*
    #define GR_ROW_BYTES(k) ((k) * GR_WQ_BITS / 8)
    #define GR_CD_RB GR_ROW_BYTES(GR_CD)
    #define GR_UP_RB GR_ROW_BYTES(GR_LOWRANK)
    // raw (unsigned domain) element e of a row, and its dequantized value as float; scale / zp are [G, N], n is the output channel
    #if GR_W_TYPE == GR_WT_U4
        #define GR_WQ_RAW(w, e) ((uint)(((w)[(e) >> 1] >> (((e) & 1) * 4)) & 0xF))
    #else
        #define GR_WQ_RAW(w, e) ((uint)(w)[e])
    #endif
    #define GR_WQ_ONE(w, scale, zp, N, GS, n, e)                                                                                  \
        convert_float(convert_half((int)GR_WQ_FLIP(GR_WQ_RAW(w, e)) - (int)GR_WQ_ZPAT(zp, (size_t)((e) / (GS)) * (N) + (n))) * \
                      as_half((scale)[(size_t)((e) / (GS)) * (N) + (n)]))
    // 16 * nv elements starting at K index k, lane-interleaved like GR_LDF (element k + 16 * v + lane in component v)
    #define GR_LDWQ_1(w, sc, zp, N, GS, n, k) GR_WQ_ONE(w, sc, zp, N, GS, n, (k) + lane)
    #define GR_LDWQ_2(w, sc, zp, N, GS, n, k) ((float2)(GR_LDWQ_1(w, sc, zp, N, GS, n, k), GR_LDWQ_1(w, sc, zp, N, GS, n, (k) + 16)))
    #define GR_LDWQ_4(w, sc, zp, N, GS, n, k) ((float4)(GR_LDWQ_1(w, sc, zp, N, GS, n, k), GR_LDWQ_1(w, sc, zp, N, GS, n, (k) + 16), \
                                                       GR_LDWQ_1(w, sc, zp, N, GS, n, (k) + 32), GR_LDWQ_1(w, sc, zp, N, GS, n, (k) + 48)))
    #define GR_LDWQ(nv, w, sc, zp, N, GS, n, k) CAT(GR_LDWQ_, nv)(w, sc, zp, N, GS, n, k)
#endif

#define GR_TH_1(r, kb) ((float)th[(r) * GR_NK + (kb)])
#define GR_TH_2(r, kb) ((float2)((float)th[(r) * GR_NK + (kb)], (float)th[(r) * GR_NK + (kb) + 1]))
#define GR_TH_4(r, kb) ((float4)((float)th[(r) * GR_NK + (kb)], (float)th[(r) * GR_NK + (kb) + 1], (float)th[(r) * GR_NK + (kb) + 2], (float)th[(r) * GR_NK + (kb) + 3]))
#define GR_TH(nv, r, kb) CAT(GR_TH_, nv)(r, kb)

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_read)(const __global ushort* residual,
                const __global ushort* norm_weight,
                GR_WPTR w_down,
                GR_WPTR w_up,
#if GR_W_TYPE != GR_WT_F16
                const __global ushort* scale_down,
                const __global ushort* scale_up,
                const __global uchar* zp_down,
                const __global uchar* zp_up,
#endif
#if GR_PRODUCE_WRITE_GATE
                GR_WPTR w_inject,
#if GR_W_TYPE != GR_WT_F16
                const __global ushort* scale_inject,
                const __global uchar* zp_inject,
#endif
#endif
                __global ushort* mixed,
#if GR_PRODUCE_WRITE_GATE
                __global ushort* write_gate,
#endif
                uint num_rows) {
    __local ushort slm_rbar[GR_READ_ROWS * GR_CD];
    __local float slm_t[GR_READ_ROWS * GR_TSTRIDE] __attribute__((aligned(16)));

    const uint row0 = get_group_id(1) * GR_READ_ROWS;
    if (row0 >= num_rows) {
        return;
    }
    const uint nrows = min((uint)GR_READ_ROWS, num_rows - row0);
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();

    // Phase 1: Rbar = rms_norm(R_c) * gain into SLM, one subgroup per (row, branch).
    for (uint p = sg; p < GR_READ_ROWS * GR_BRANCHES; p += GR_WG_SIZE) {
        const uint r = p / GR_BRANCHES;
        const uint c = p % GR_BRANCHES;
        if (r >= nrows) {
            continue;
        }
        const size_t gbase = (size_t)(row0 + r) * GR_CD + c * GR_HIDDEN;
        const uint sbase = r * GR_CD + c * GR_HIDDEN;
        float sq = 0.0f;
        for (uint d = 0; d < GR_HIDDEN; d += GR_KD) {
            const GR_UT(GR_NV_D) xu = GR_BRU(GR_NV_D, residual + gbase + d);
            const GR_FT(GR_NV_D) f = GR_U2F(GR_NV_D, xu);
            const GR_FT(GR_NV_D) q = f * f;
            GR_ACC(GR_NV_D, sq, q);
            GR_SLMST(GR_NV_D, slm_rbar + sbase + d, xu, lane);
        }
        const float inv_rms = native_rsqrt(sub_group_reduce_add(sq) * (1.0f / (float)GR_HIDDEN) + GR_EPS);
        for (uint d = 0; d < GR_HIDDEN; d += GR_KD) {
            const GR_UT(GR_NV_D) xu = GR_SLMLD(GR_NV_D, slm_rbar + sbase + d, lane);
            const GR_UT(GR_NV_D) gu = GR_BRU(GR_NV_D, norm_weight + c * GR_HIDDEN + d);
            const GR_FT(GR_NV_D) f = GR_U2F(GR_NV_D, xu);
            const GR_FT(GR_NV_D) gain = GR_U2F(GR_NV_D, gu);
            const GR_FT(GR_NV_D) o = (f * inv_rms) * gain;
            const GR_UT(GR_NV_D) ou = GR_F2U(GR_NV_D, o);
            GR_SLMST(GR_NV_D, slm_rbar + sbase + d, ou, lane);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: low-rank down projection (+ write-gate rows), GR_NJ outputs x GR_READ_ROWS rows per pass.
    for (uint l0 = sg * GR_NJ; l0 < GR_NOUT; l0 += GR_WG_SIZE * GR_NJ) {
        float acc[GR_NJ * GR_READ_ROWS];
#pragma unroll
        for (uint i = 0; i < GR_NJ * GR_READ_ROWS; i++) {
            acc[i] = 0.0f;
        }
        for (uint k = 0; k < GR_CD; k += GR_KC) {
            GR_FT(GR_NV_C) rv[GR_READ_ROWS];
#pragma unroll
            for (uint r = 0; r < GR_READ_ROWS; r++) {
                rv[r] = GR_U2F(GR_NV_C, GR_SLMLD(GR_NV_C, slm_rbar + r * GR_CD + k, lane));
            }
#pragma unroll
            for (uint j = 0; j < GR_NJ; j++) {
                const uint l = min(l0 + j, (uint)(GR_NOUT - 1));
#if GR_W_TYPE == GR_WT_F16
#if GR_PRODUCE_WRITE_GATE
                const __global ushort* wp = l >= GR_LOWRANK ? w_inject + (size_t)(l - GR_LOWRANK) * GR_CD : w_down + (size_t)l * GR_CD;
#else
                const __global ushort* wp = w_down + (size_t)l * GR_CD;
#endif
                const GR_FT(GR_NV_C) wf = GR_LDF(GR_NV_C, wp + k);
#else
#if GR_PRODUCE_WRITE_GATE
                const bool gate_row = l >= GR_LOWRANK;
                GR_WPTR wp = gate_row ? w_inject + (size_t)(l - GR_LOWRANK) * GR_CD_RB : w_down + (size_t)l * GR_CD_RB;
                const __global ushort* wsc = gate_row ? scale_inject : scale_down;
                const __global uchar* wzp = gate_row ? zp_inject : zp_down;
                const uint wn = gate_row ? l - GR_LOWRANK : l;
                const uint wN = gate_row ? GR_BRANCHES : GR_LOWRANK;
                const uint wgs = gate_row ? GR_RD_GSI : GR_RD_GSD;
#else
                GR_WPTR wp = w_down + (size_t)l * GR_CD_RB;
                const __global ushort* wsc = scale_down;
                const __global uchar* wzp = zp_down;
                const uint wn = l;
                const uint wN = GR_LOWRANK;
                const uint wgs = GR_RD_GSD;
#endif
                const GR_FT(GR_NV_C) wf = GR_LDWQ(GR_NV_C, wp, wsc, wzp, wN, wgs, wn, k);
#endif
#pragma unroll
                for (uint r = 0; r < GR_READ_ROWS; r++) {
                    GR_DOT(GR_NV_C, acc[j * GR_READ_ROWS + r], wf, rv[r]);
                }
            }
        }
#pragma unroll
        for (uint r = 0; r < GR_READ_ROWS; r++) {
            float4 logits;
            logits.s0 = sub_group_reduce_add(acc[0 * GR_READ_ROWS + r]) * GR_INV_C;
            logits.s1 = sub_group_reduce_add(acc[1 * GR_READ_ROWS + r]) * GR_INV_C;
            logits.s2 = sub_group_reduce_add(acc[2 * GR_READ_ROWS + r]) * GR_INV_C;
            logits.s3 = sub_group_reduce_add(acc[3 * GR_READ_ROWS + r]) * GR_INV_C;
            const float4 sig = GR_SIGMOID(logits);
            float4 act = logits * sig;
#if GR_PRODUCE_WRITE_GATE
            // The write gate lives in (0, 2); it must not be clamped to (0, 1).
            if (l0 >= GR_LOWRANK) {
                act = 2.0f * sig;
            }
#endif
            if (lane == 0) {
                vstore4(act, 0, slm_t + r * GR_TSTRIDE + l0);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if GR_PRODUCE_WRITE_GATE
    if (sg < nrows && lane < GR_BRANCHES) {
        write_gate[(size_t)(row0 + sg) * GR_BRANCHES + lane] = as_ushort(convert_half_rte(slm_t[sg * GR_TSTRIDE + GR_LOWRANK + lane]));
    }
#endif

    // Phase 3: per-(branch, channel) sigmoid read gate and branch reduce mean. t is held as half in registers.
    half th[GR_READ_ROWS * GR_NK];
#pragma unroll
    for (uint r = 0; r < GR_READ_ROWS; r++) {
#pragma unroll
        for (uint kb = 0; kb < GR_NK; kb++) {
            th[r * GR_NK + kb] = convert_half_rte(slm_t[r * GR_TSTRIDE + kb * 16 + lane]);
        }
    }

    for (uint d0 = sg * 16; d0 < GR_HIDDEN; d0 += GR_WG_SIZE * 16) {
        float mix[GR_READ_ROWS];
#pragma unroll
        for (uint r = 0; r < GR_READ_ROWS; r++) {
            mix[r] = 0.0f;
        }
        for (uint c = 0; c < GR_BRANCHES; c++) {
            float logit[GR_READ_ROWS];
#pragma unroll
            for (uint j = 0; j < 16; j++) {
                float a[GR_READ_ROWS];
#pragma unroll
                for (uint r = 0; r < GR_READ_ROWS; r++) {
                    a[r] = 0.0f;
                }
#if GR_W_TYPE == GR_WT_F16
                const __global ushort* wrow = w_up + (size_t)(c * GR_HIDDEN + d0 + j) * GR_LOWRANK;
#else
                GR_WPTR wrow = w_up + (size_t)(c * GR_HIDDEN + d0 + j) * GR_UP_RB;
#endif
#pragma unroll
                for (uint k = 0; k < GR_LOWRANK; k += GR_KL) {
#if GR_W_TYPE == GR_WT_F16
                    const GR_FT(GR_NV_L) wf = GR_LDF(GR_NV_L, wrow + k);
#else
                    const GR_FT(GR_NV_L) wf = GR_LDWQ(GR_NV_L, wrow, scale_up, zp_up, GR_CD, GR_RD_GSU, c * GR_HIDDEN + d0 + j, k);
#endif
#pragma unroll
                    for (uint r = 0; r < GR_READ_ROWS; r++) {
                        const GR_FT(GR_NV_L) tv = GR_TH(GR_NV_L, r, k / 16);
                        GR_DOT(GR_NV_L, a[r], wf, tv);
                    }
                }
#pragma unroll
                for (uint r = 0; r < GR_READ_ROWS; r++) {
                    const float total = sub_group_reduce_add(a[r]);
                    logit[r] = j == 0 ? (lane == 0 ? total : 0.0f) : (lane == j ? total : logit[r]);
                }
            }
#pragma unroll
            for (uint r = 0; r < GR_READ_ROWS; r++) {
                const float rbar = convert_float(as_half(slm_rbar[r * GR_CD + c * GR_HIDDEN + d0 + lane]));
                mix[r] += GR_SIGMOID(logit[r]) * rbar;
            }
        }
#pragma unroll
        for (uint r = 0; r < GR_READ_ROWS; r++) {
            if (r < nrows) {
                intel_sub_group_block_write_us(mixed + (size_t)(row0 + r) * GR_HIDDEN + d0, as_ushort(convert_half_rte(mix[r] * GR_INV_C)));
            }
        }
    }
}
