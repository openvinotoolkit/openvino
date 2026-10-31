// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GatedResidual READ / MIX as ONE systolic (dpas) kernel. One work-group owns GR_RD_TM token rows:
//   phase 1  inv_rms[row, c] of the branch-wise RMSNorm                       -> SLM
//   phase 2  t = SiLU(Rbar @ W_down^T / C) (+ write gate S as an extra tile)  -> SLM (dpas A-operand slots)
//   phase 3  mixed = (1/C) sum_c sigmoid(t @ W_up^T)[c] * Rbar[c]             -> global
// Rbar is never materialized: phase 2 scales the raw residual by the gain in half precision and applies inv_rms to the
// f32 accumulator once per branch; phase 3 recomputes Rbar in f32. The accumulator layout of a dpas (token i in
// component i, column n in lane n) is already the A-operand layout of the next one, so t only needs an SLM slot
// of 8 tokens x 16 low-rank values per (token tile, K16 block).
// The residual tiles come through 2D block reads (a whole TM x K32 A tile per message, rows past the end read 0); W rows
// are K-contiguous, so every lane reads its own row with one 64 B vload16.
//
// JIT: GR_BRANCHES, GR_HIDDEN (multiple of 32), GR_LOWRANK (multiple of 32), GR_EPS, GR_NORM_GAIN_PLUS_ONE,
//      GR_PRODUCE_WRITE_GATE, GR_WG_SIZE (subgroups), GR_RD_TM (8/16/32), GR_RD_NT, GR_NV_KD (phase 1 chunk halves / 16),
//      GR_RD_SLM_GAIN (0 = gain from global, 1 = padded SLM copy, 2 = duplicated SLM copy), GR_P3_GB (optional: branches per
//      phase-3 K loop).
// Compressed weights (GR_W_TYPE u4 group 32 / u8 per-channel, gr_wq.cl; no write gate): GR_WD_KCH / GR_WD_NRAW / GR_WD_RB and
//      GR_WU_KCH / GR_WU_NRAW / GR_WU_RB give the k values per raw load, its dwords and the row bytes of W_down / W_up.
// Dispatch: gws = [WG * 16, ceil(rows / TM), 1], lws = [WG * 16, 1, 1].

#include "gr_common.cl"
#include "gr_wq.cl"

#define GR_CD (GR_BRANCHES * GR_HIDDEN)
#define GR_MT (GR_RD_TM / 8)
#define GR_KB (GR_LOWRANK / 16)
#define GR_TASKS (GR_KB / GR_RD_NT + (GR_PRODUCE_WRITE_GATE ? 1 : 0))
#define GR_KD (GR_NV_KD * 16)
#define GR_INV_C (1.0f / (float)GR_BRANCHES)

#if GR_HIDDEN % 32 != 0 || GR_LOWRANK % 32 != 0 || GR_KB % GR_RD_NT != 0
    #error "gr_read_dpas.cl: hidden size and low rank must be multiples of 32 and GR_RD_NT must divide L / 16"
#endif
#if GR_BRANCHES % 2 != 0 || GR_BRANCHES > 16
    #error "gr_read_dpas.cl: the write gate is one N16 tile stored in half pairs"
#endif
#if GR_W_TYPE != GR_WT_F16 && GR_PRODUCE_WRITE_GATE
    #error "gr_read_dpas.cl: compressed weights do not support the write gate"
#endif
#if GR_W_TYPE != GR_WT_F16 && (GR_HIDDEN % GR_WD_KCH != 0 || GR_LOWRANK % GR_WU_KCH != 0)
    #error "gr_read_dpas.cl: D and L must be multiples of the weight load widths"
#endif

// The 1 of the (1 + w) gain is added in half precision
#if GR_NORM_GAIN_PLUS_ONE
    #define GR_GADD(g) ((g) + (half)1.0f)
#else
    #define GR_GADD(g) (g)
#endif

#define GR_P2_LDB(dst, j, KK) dst = vload16(0, (const __global uint*)(brow[j] + (KK)))

// The gain of a K16 slice is one register half: A rows alternate between the low and high half of a GRF, and IGC realigns
// (mov + mul + mov) whenever the gain sits at the other offset. The gain (+1) is therefore staged in SLM so that every gain
// half is also read at the other offset: duplicated (GS = (g0, g0, g1, g1)) or behind a 16-half pad (GE = (g0, g1),
// GS = (g-1, g0, g1, g2)).
#if GR_RD_SLM_GAIN == 2
    #define GR_SLM_G_SIZE (2 * GR_CD)
    #define GR_P2_GAIN(GE, GS, KK) { GS = BLOCK_READN_SLM(ushort, 4, slm_g + 2 * (KK), 0); }
    #define GR_G0E(GE, GS) as_half((GS).s0)
    #define GR_G0O(GE, GS) as_half((GS).s1)
    #define GR_G1E(GE, GS) as_half((GS).s2)
    #define GR_G1O(GE, GS) as_half((GS).s3)
#elif GR_RD_SLM_GAIN
    #define GR_SLM_G_SIZE (GR_CD + 48)
    #define GR_P2_GAIN(GE, GS, KK) { GE = BLOCK_READN_SLM(ushort, 2, slm_g + 16 + (KK), 0); GS = BLOCK_READN_SLM(ushort, 4, slm_g + (KK), 0); }
    #define GR_G0E(GE, GS) as_half((GE).s0)
    #define GR_G0O(GE, GS) as_half((GS).s1)
    #define GR_G1E(GE, GS) as_half((GS).s2)
    #define GR_G1O(GE, GS) as_half((GE).s1)
#else
    #define GR_P2_GAIN(GE, GS, KK) { GE = intel_sub_group_block_read_us2(norm_weight + (KK)); }
    #define GR_G0E(GE, GS) GR_GADD(as_half((GE).s0))
    #define GR_G0O(GE, GS) GR_GADD(as_half((GE).s0))
    #define GR_G1E(GE, GS) GR_GADD(as_half((GE).s1))
    #define GR_G1O(GE, GS) GR_GADD(as_half((GE).s1))
#endif

// One K32 step of phase 2: loads of the A tile (raw residual), the gain and the W rows of every column tile ...
#define GR_P2_LOAD(AT, GE, GS, B, KK) { GR_RD2D(GR_RD_TM, residual, GR_CD * 2, num_rows, GR_CD * 2, (KK), t0, AT); GR_P2_GAIN(GE, GS, KK); unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) { GR_P2_LDB(B[j_], j_, KK); } }

#define GR_P2_SCALE(v, ge, go) as_short8(as_half8(v) * (half8)(ge, go, ge, go, ge, go, ge, go))

// ... and its dpas: A = half(R * gain) (rounded in half precision), two K16 slices, slice 0 first
#define GR_P2_MMA(AT, GE, GS, B)                                                                                                   \
    {                                                                                                                              \
        const half g0e_ = GR_G0E(GE, GS);                                                                                          \
        const half g0o_ = GR_G0O(GE, GS);                                                                                          \
        const half g1e_ = GR_G1E(GE, GS);                                                                                          \
        const half g1o_ = GR_G1O(GE, GS);                                                                                          \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                                               \
            const short8 av_ = GR_P2_SCALE(vload8(m_, (AT)), g0e_, g0o_);                                                          \
            unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) {                                                                        \
                accc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].lo), accc[j_ * GR_MT + m_]);                                   \
            }                                                                                                                      \
        }                                                                                                                          \
        unroll_for (uint m_ = 0; m_ < GR_MT; m_++) {                                                                               \
            const short8 av_ = GR_P2_SCALE(vload8(GR_MT + m_, (AT)), g1e_, g1o_);                                                  \
            unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) {                                                                        \
                accc[j_ * GR_MT + m_] = GR_MAD(av_, as_int8((B)[j_].hi), accc[j_ * GR_MT + m_]);                                   \
            }                                                                                                                      \
        }                                                                                                                          \
    }

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_read_dpas)(const __global ushort* residual,
                     const __global ushort* norm_weight,
                     const __global uchar* w_down,
                     const __global uchar* w_up,
                     const __global ushort* scale_down,
                     const __global ushort* scale_up,
                     const __global uchar* zp_down,
                     const __global uchar* zp_up,
#if GR_PRODUCE_WRITE_GATE
                     const __global ushort* w_inject,
#endif
                     __global ushort* mixed,
#if GR_PRODUCE_WRITE_GATE
                     __global ushort* write_gate,
#endif
                     uint num_rows) {
    __local float slm_inv[GR_BRANCHES * GR_RD_TM] __attribute__((aligned(32)));
    __local ushort slm_t[GR_MT * GR_KB * 128] __attribute__((aligned(16)));
#if GR_RD_SLM_GAIN
    __local ushort slm_g[GR_SLM_G_SIZE] __attribute__((aligned(64)));
#endif

    const uint lane = get_sub_group_local_id();
    const uint sg = get_sub_group_id();
    const uint t0 = get_group_id(1) * GR_RD_TM;
    if (t0 >= num_rows) {
        return;
    }

#if GR_RD_SLM_GAIN
    {
        // All chunk loads are issued before the SLM stores so the copy costs one load latency.
#if GR_CD % 128 == 0
    #define GR_GNV 8
#else
    #define GR_GNV 2
#endif
#define GR_GIT ((GR_CD + GR_WG_SIZE * GR_GNV * 16 - 1) / (GR_WG_SIZE * GR_GNV * 16))
        GR_UT(GR_GNV) gv[GR_GIT];
#pragma unroll
        for (uint it = 0; it < GR_GIT; it++) {
            gv[it] = GR_BRU(GR_GNV, norm_weight + min((it * GR_WG_SIZE + sg) * GR_GNV * 16, (uint)(GR_CD - GR_GNV * 16)));
        }
#pragma unroll
        for (uint it = 0; it < GR_GIT; it++) {
            const uint i = (it * GR_WG_SIZE + sg) * GR_GNV * 16;
            if (i < GR_CD) {
                const GR_UT(GR_GNV) g = CAT(GR_ASU_, GR_GNV)(GR_GADD(CAT(GR_ASH_, GR_GNV)(gv[it])));
#if GR_RD_SLM_GAIN == 2
                ushort gq[GR_GNV];
                CAT(vstore, GR_GNV)(g, 0, gq);
#pragma unroll
                for (uint q = 0; q < GR_GNV; q++) {
                    BLOCK_WRITEN_SLM(ushort, 2, slm_g + 2 * (i + 16 * q), 0, (ushort2)(gq[q], gq[q]));
                }
#else
                BLOCK_WRITEN_SLM(ushort, GR_GNV, slm_g + 16 + i, 0, g);
#endif
            }
        }
    }
#endif

    // Phase 1: inv_rms[c][row] for 4 rows per task. Rows past num_rows re-read the last row and get 0.
    for (uint p = sg; p < GR_BRANCHES * (GR_RD_TM / 4); p += GR_WG_SIZE) {
        const uint c = p / (GR_RD_TM / 4);
        const uint r0 = (p % (GR_RD_TM / 4)) * 4;
        size_t off[4];
        float sq[4];
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            off[r] = (size_t)min(t0 + r0 + r, num_rows - 1) * GR_CD + c * GR_HIDDEN;
            sq[r] = 0.0f;
        }
        // The next chunk of the 4 rows is in flight while the current one is reduced (the last step re-reads the last chunk).
        GR_UT(GR_NV_KD) x[4];
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            x[r] = GR_BRU(GR_NV_KD, residual + off[r]);
        }
        for (uint d = 0; d < GR_HIDDEN; d += GR_KD) {
            const uint dn = min(d + GR_KD, (uint)(GR_HIDDEN - GR_KD));
            GR_UT(GR_NV_KD) xn[4];
#pragma unroll
            for (uint r = 0; r < 4; r++) {
                xn[r] = GR_BRU(GR_NV_KD, residual + off[r] + dn);
            }
#pragma unroll
            for (uint r = 0; r < 4; r++) {
                const GR_FT(GR_NV_KD) f = GR_U2F(GR_NV_KD, x[r]);
                const GR_FT(GR_NV_KD) q = f * f;
                GR_ACC(GR_NV_KD, sq[r], q);
                x[r] = xn[r];
            }
        }
        float4 inv;
        inv.s0 = t0 + r0 + 0 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[0]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        inv.s1 = t0 + r0 + 1 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[1]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        inv.s2 = t0 + r0 + 2 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[2]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        inv.s3 = t0 + r0 + 3 < num_rows ? native_rsqrt(sub_group_reduce_add(sq[3]) * (1.0f / (float)GR_HIDDEN) + GR_EPS) : 0.0f;
        if (lane == 0) {
            vstore4(inv, 0, slm_inv + c * GR_RD_TM + r0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: down projection, GR_RD_NT low-rank tiles (or the write-gate tile) per task.
    //   acc += inv_rms[:, c] * ((R[:, c] * gain[c]) @ W[:, c]^T)   per branch c
    for (uint task = sg; task < GR_TASKS; task += GR_WG_SIZE) {
#if GR_PRODUCE_WRITE_GATE
        const bool is_wgate = task == GR_TASKS - 1;
        const __global ushort* w = is_wgate ? w_inject : w_down;
        const uint n0 = is_wgate ? 0 : task * GR_RD_NT * 16;
        const uint w_rows = is_wgate ? GR_BRANCHES : GR_LOWRANK;
#else
        const uint n0 = task * GR_RD_NT * 16;
#endif
        const __global uchar* brow[GR_RD_NT];
#pragma unroll
        for (uint j = 0; j < GR_RD_NT; j++) {
#if GR_PRODUCE_WRITE_GATE
            const uint row = is_wgate ? min(lane, (uint)(GR_BRANCHES - 1)) : n0 + 16 * j + lane;
#else
            const uint row = n0 + 16 * j + lane;
#endif
            brow[j] = w_down + (size_t)row * GR_WD_RB;
        }
#if GR_W_TYPE == GR_WT_U8
        // per-channel: one scale / zp per output channel for the whole K
        half2 scv[GR_RD_NT];
        half2 zpv[GR_RD_NT];
        unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) {
            scv[j_] = GR_WQ_SCALE2(scale_down, 0, GR_LOWRANK, n0 + 16 * j_);
            zpv[j_] = GR_WQ_ZP2(zp_down, 0, GR_LOWRANK, n0 + 16 * j_, lane);
        }
#endif

        float8 acc[GR_RD_NT * GR_MT];
#pragma unroll
        for (uint i = 0; i < GR_RD_NT * GR_MT; i++) {
            acc[i] = (float8)(0.0f);
        }
        for (uint c = 0; c < GR_BRANCHES; c++) {
            float8 accc[GR_RD_NT * GR_MT];
#pragma unroll
            for (uint i = 0; i < GR_RD_NT * GR_MT; i++) {
                accc[i] = (float8)(0.0f);
            }
            const uint kbeg = c * GR_HIDDEN;
            const uint kend = kbeg + GR_HIDDEN;
            ushort at0[2 * GR_RD_TM];
            ushort2 ge0;
            ushort4 gs0;
            uint16 b0[GR_RD_NT];
            for (uint k = kbeg; k < kend; k += GR_WD_KCH) {
                uint rw[GR_RD_NT][GR_WD_NRAW];
                unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) {
                    GR_WQ_LOAD_RAW(rw[j_], GR_WD_NRAW, brow[j_], k);
                }
                unroll_for (uint s_ = 0; s_ < GR_WD_KCH / 32; s_++) {
                    GR_RD2D(GR_RD_TM, residual, GR_CD * 2, num_rows, GR_CD * 2, k + 32 * s_, t0, at0);
                    GR_P2_GAIN(ge0, gs0, k + 32 * s_);
                    unroll_for (uint j_ = 0; j_ < GR_RD_NT; j_++) {
#if GR_W_TYPE == GR_WT_U4
                        const half2 scv = GR_WQ_SCALE2(scale_down, (k >> 5) + s_, GR_LOWRANK, n0 + 16 * j_);
                        const half2 zpv = GR_WQ_ZP2(zp_down, (k >> 5) + s_, GR_LOWRANK, n0 + 16 * j_, lane);
                        b0[j_] = GR_WQ_DEQ32(rw[j_], s_ * GR_WQ_RAW_PER_K32, zpv, scv);
#else
                        b0[j_] = GR_WQ_DEQ32(rw[j_], s_ * GR_WQ_RAW_PER_K32, zpv[j_], scv[j_]);
#endif
                    }
                    GR_P2_MMA(at0, ge0, gs0, b0);
                }
            }
            float8 invv[GR_MT];
#pragma unroll
            for (uint m = 0; m < GR_MT; m++) {
                invv[m] = vload8(0, slm_inv + c * GR_RD_TM + m * 8);
            }
#pragma unroll
            for (uint m = 0; m < GR_MT; m++) {
#pragma unroll
                for (uint j = 0; j < GR_RD_NT; j++) {
                    acc[j * GR_MT + m] += accc[j * GR_MT + m] * invv[m];
                }
            }
        }

        const float inv_c = GR_INV_C;
#if GR_PRODUCE_WRITE_GATE
        if (is_wgate) {
            if (lane < GR_BRANCHES) {
#pragma unroll
                for (uint m = 0; m < GR_MT; m++) {
                    // Write-gate range is (0, 2) - it must not be clamped to (0, 1).
                    const float8 s = 2.0f * GR_SIGMOID(acc[m] * inv_c);
                    float tmp[8];
                    vstore8(s, 0, tmp);
#pragma unroll
                    for (uint i = 0; i < 8; i++) {
                        const uint tok = t0 + m * 8 + i;
                        if (tok < num_rows) {
                            write_gate[tok * GR_BRANCHES + lane] = as_ushort(convert_half_rte(tmp[i]));
                        }
                    }
                }
            }
            continue;
        }
#endif
#pragma unroll
        for (uint j = 0; j < GR_RD_NT; j++) {
#pragma unroll
            for (uint m = 0; m < GR_MT; m++) {
                const float8 x = acc[j * GR_MT + m] * inv_c;
                const ushort8 tu = as_ushort8(convert_half8_rte(x * GR_SIGMOID(x)));
                BLOCK_WRITEN_SLM(ushort, 8, slm_t + (m * GR_KB + n0 / 16 + j) * 128, 0, tu);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: up projection + sigmoid read gate + branch mix, 16 channels per pass. Up to 4 branches run in one K loop so
    // each SLM read of t feeds several dpas; the mix still adds the branches in order 0..C-1.
#ifndef GR_P3_GB
#define GR_P3_GB (GR_BRANCHES % 4 == 0 ? 4 : 2)
#endif
    for (uint dt = sg; dt < GR_HIDDEN / 16; dt += GR_WG_SIZE) {
        const uint d0 = dt * 16;
        float8 mix[GR_MT];
#pragma unroll
        for (uint m = 0; m < GR_MT; m++) {
            mix[m] = (float8)(0.0f);
        }
        for (uint cg = 0; cg < GR_BRANCHES; cg += GR_P3_GB) {
            float8 acc[GR_P3_GB * GR_MT];
#pragma unroll
            for (uint i = 0; i < GR_P3_GB * GR_MT; i++) {
                acc[i] = (float8)(0.0f);
            }
            const __global uchar* wrow[GR_P3_GB];
#pragma unroll
            for (uint g = 0; g < GR_P3_GB; g++) {
                wrow[g] = w_up + (size_t)((cg + g) * GR_HIDDEN + d0 + lane) * GR_WU_RB;
            }
#if GR_W_TYPE == GR_WT_U8
            half2 scv[GR_P3_GB];
            half2 zpv[GR_P3_GB];
            unroll_for (uint g_ = 0; g_ < GR_P3_GB; g_++) {
                scv[g_] = GR_WQ_SCALE2(scale_up, 0, GR_CD, (cg + g_) * GR_HIDDEN + d0);
                zpv[g_] = GR_WQ_ZP2(zp_up, 0, GR_CD, (cg + g_) * GR_HIDDEN + d0, lane);
            }
#endif
            for (uint kb = 0; kb < GR_KB; kb += GR_WU_KCH / 16) {
                uint rw[GR_P3_GB][GR_WU_NRAW];
                unroll_for (uint g_ = 0; g_ < GR_P3_GB; g_++) {
                    GR_WQ_LOAD_RAW(rw[g_], GR_WU_NRAW, wrow[g_], kb * 16);
                }
                unroll_for (uint s_ = 0; s_ < GR_WU_KCH / 32; s_++) {
                    uint16 b[GR_P3_GB];
                    unroll_for (uint g_ = 0; g_ < GR_P3_GB; g_++) {
#if GR_W_TYPE == GR_WT_U4
                        const half2 scv = GR_WQ_SCALE2(scale_up, (kb >> 1) + s_, GR_CD, (cg + g_) * GR_HIDDEN + d0);
                        const half2 zpv = GR_WQ_ZP2(zp_up, (kb >> 1) + s_, GR_CD, (cg + g_) * GR_HIDDEN + d0, lane);
                        b[g_] = GR_WQ_DEQ32(rw[g_], s_ * GR_WQ_RAW_PER_K32, zpv, scv);
#else
                        b[g_] = GR_WQ_DEQ32(rw[g_], s_ * GR_WQ_RAW_PER_K32, zpv[g_], scv[g_]);
#endif
                    }
#pragma unroll
                    for (uint m = 0; m < GR_MT; m++) {
                        const short8 a0 = as_short8(BLOCK_READN_SLM(ushort, 8, slm_t + (m * GR_KB + kb + 2 * s_) * 128, 0));
                        const short8 a1 = as_short8(BLOCK_READN_SLM(ushort, 8, slm_t + (m * GR_KB + kb + 2 * s_ + 1) * 128, 0));
#pragma unroll
                        for (uint g = 0; g < GR_P3_GB; g++) {
                            acc[g * GR_MT + m] = GR_MAD(a0, as_int8(b[g].lo), acc[g * GR_MT + m]);
                            acc[g * GR_MT + m] = GR_MAD(a1, as_int8(b[g].hi), acc[g * GR_MT + m]);
                        }
                    }
                }
            }
#pragma unroll
            for (uint g = 0; g < GR_P3_GB; g++) {
                const uint c = cg + g;
                ushort rt[GR_MT][8];
#pragma unroll
                for (uint m = 0; m < GR_MT; m++) {
                    GR_RD2D1(8, residual, GR_CD * 2, num_rows, GR_CD * 2, c * GR_HIDDEN + d0, t0 + m * 8, rt[m]);
                }
                float gain = convert_float(as_half(intel_sub_group_block_read_us(norm_weight + c * GR_HIDDEN + d0)));
#if GR_NORM_GAIN_PLUS_ONE
                gain = gain + 1.0f;
#endif
#pragma unroll
                for (uint m = 0; m < GR_MT; m++) {
                    const float8 invv = vload8(0, slm_inv + c * GR_RD_TM + m * 8);
                    const float8 rbar = (convert_float8(as_half8(vload8(0, rt[m]))) * gain) * invv;
                    mix[m] += GR_SIGMOID(acc[g * GR_MT + m]) * rbar;
                }
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
                if (tok < num_rows) {
                    intel_sub_group_block_write_us(mixed + tok * GR_HIDDEN + d0, tmp[i]);
                }
            }
        }
    }
}
