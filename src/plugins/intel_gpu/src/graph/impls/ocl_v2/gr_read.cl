// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GatedResidual READ / MIX (also FINAL_MIX), fused scalar fallback (pre-Xe2 or shapes the systolic path rejects).
//
//   Rbar = branch-wise RMSNorm(R) * gain  ->  t = SiLU(W_down Rbar / C)  ->  G = sigmoid(W_up t)
//   mixed = (1/C) * sum_c G_c * Rbar_c,   optionally S = 2 * sigmoid(W_inject Rbar / C)
//
// One work-group owns GR_READ_ROWS token rows; one subgroup is one hardware thread. Rbar and t live in SLM.
// JIT: GR_BRANCHES (4), GR_HIDDEN, GR_LOWRANK, GR_EPS, GR_NORM_GAIN_PLUS_ONE, GR_PRODUCE_WRITE_GATE,
//      GR_WG_SIZE (subgroups per work-group), GR_READ_ROWS, GR_NV_D / GR_NV_C / GR_NV_L (= chunk halves / 16 of the
//      three K loops over D, C*D and L).
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

#define GR_TH_1(r, kb) ((float)th[(r) * GR_NK + (kb)])
#define GR_TH_2(r, kb) ((float2)((float)th[(r) * GR_NK + (kb)], (float)th[(r) * GR_NK + (kb) + 1]))
#define GR_TH_4(r, kb) ((float4)((float)th[(r) * GR_NK + (kb)], (float)th[(r) * GR_NK + (kb) + 1], (float)th[(r) * GR_NK + (kb) + 2], (float)th[(r) * GR_NK + (kb) + 3]))
#define GR_TH(nv, r, kb) CAT(GR_TH_, nv)(r, kb)

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_read)(const __global ushort* residual,
                const __global ushort* norm_weight,
                const __global ushort* w_down,
                const __global ushort* w_up,
#if GR_PRODUCE_WRITE_GATE
                const __global ushort* w_inject,
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
            GR_FT(GR_NV_D) gain = GR_U2F(GR_NV_D, gu);
#if GR_NORM_GAIN_PLUS_ONE
            gain = gain + 1.0f;
#endif
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
#if GR_PRODUCE_WRITE_GATE
                const __global ushort* wp = l >= GR_LOWRANK ? w_inject + (size_t)(l - GR_LOWRANK) * GR_CD : w_down + (size_t)l * GR_CD;
#else
                const __global ushort* wp = w_down + (size_t)l * GR_CD;
#endif
                const GR_FT(GR_NV_C) wf = GR_LDF(GR_NV_C, wp + k);
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
                const __global ushort* wrow = w_up + (size_t)(c * GR_HIDDEN + d0 + j) * GR_LOWRANK;
#pragma unroll
                for (uint k = 0; k < GR_LOWRANK; k += GR_KL) {
                    const GR_FT(GR_NV_L) wf = GR_LDF(GR_NV_L, wrow + k);
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
