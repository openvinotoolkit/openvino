// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// GatherMatmul prefill on int8 DPAS for base-3 ternary (t1) and binary (u1) expert weights, scale groups of 128.
// A: [n_act, n_tokens, K] f16 (n_act == 1 broadcasts one row to all slots); indices: [n_tokens, top_k] expert ids;
// out: [top_k, n_tokens, N] f16. The (token, slot) rows are routed expert-major into "packed rows" (only an index
// map: the quantize stages read A through it and the tile stage stores through it). One file, seven kernels (STAGE):
//   1 zero  : counts[E] = 0
//   2 count : per (token, slot): pos = atomic_inc(counts[expert])
//   3 build : one work-item: offsets[E], 8-row subtiles {first packed row, live rows} and tiles {expert, first subtile,
//             subtiles <= SUBT}; meta = {subtiles, tiles}
//   4 place : per (token, slot): packed row offsets[expert] + pos -> row_token, row_slot
//   5 quant_scale: per (packed row, 128-group) int8 scale and code sum
//   6 quant_dpas : activations quantized into the DPAS A operand order of each subtile
//   7 tile       : per (tile, 128 output columns): each lane decodes its column's weights once per unit into DPAS B
//                  operands and runs them against the tile's subtiles
// The order of rows inside an expert depends on the atomics; every row is computed on its own, so the output does not.
// Operand order per unit (the digit/bit extraction of the dp4a kernel):
//   t1 (256 values, 13 uints): group A = (k 0..4) x (uint 0..5), then (k 0..4) x (uint 6 & 0x0000FFFF) = 35 -> 40;
//                              group B = (k 0..4) x (uint 6 & 0xFFFF0000), then (k 0..4) x (uint 7..12) = 35 -> 40
//   u1 (128 values, 4 uints) : (bit k 0..7) x (uint 0..3) = 32
// Weights: [E, N, K / 64 * 13] (t1) or [E, N, K / 8] (u1) bytes; scales [E, K / 128, N] (f16, or FP8 E4M3 bytes).

#define SIMD 16
// One quantizer for both stages: STAGE 5 sums the codes into XSUM, STAGE 6 packs the same codes for DPAS, so they
// must round identically (same expression, x * (1 / scale) round-to-nearest-even, clamped to the int8 range used).
#define QUANT_A(x, scale) clamp(convert_int_rte((float)(x) * (1.f / (scale))), -127, 127)
#define G 128
#define NG (K_SIZE / G)
#if WEIGHTS_U1
#    define UNIT_VALUES 128
#    define NCH 4
#    define ZP_F U1_ZP
#    define ROW_BYTES (K_SIZE / 8)
#    define GPU_ 1
#else
#    define UNIT_VALUES 256
#    define NCH 10
#    define ZP_F 1.0f
#    define ROW_BYTES (K_SIZE / 64 * 13)
#    define GPU_ 2
#endif
#define UNITS (K_SIZE / UNIT_VALUES)
#define CHB (8 * 16)
#define WG_SG 8

#if SCALES_FP8
#    define SCALE_T uchar
#    define SCALE_VAL(c) ((float)as_half((ushort)((((uint)(c) & 0x80u) << 8) | (((uint)(c) & 0x7Fu) << 7))) * 256.0f)
#else
#    define SCALE_T half
#    define SCALE_VAL(h) ((float)(h))
#endif

// the input row of a packed row (n_act == 1 broadcasts one row to all slots)
#define SRC_ROW(prow) ((size_t)(min(row_slot[(prow)], n_act - 1) * n_tokens + row_token[(prow)]))


#if !WEIGHTS_U1
// operand list entry for chunk c, slot j (0..7): (k, i, mask); pads -> mask 0
inline void FUNC(t1_op)(int c, int j, int* k, int* i, uint* mask) {
    const int grp = c / 5, o = (c % 5) * 8 + j;
    *mask = 0xFFFFFFFFu;
    if (o >= 35) {
        *k = 0;
        *i = 0;
        *mask = 0u;
        return;
    }
    if (grp == 0) {
        if (o < 30) {
            *k = o / 6;
            *i = o % 6;
        } else {
            *k = o - 30;
            *i = 6;
            *mask = 0x0000FFFFu;
        }
    } else {
        if (o < 5) {
            *k = o;
            *i = 6;
            *mask = 0xFFFF0000u;
        } else {
            *k = (o - 5) / 6;
            *i = 7 + (o - 5) % 6;
        }
    }
}
#endif

#if STAGE == 1
KERNEL(gathermatmul_t1_dpas)(__global int* counts) {
    const uint e = get_global_id(0);
    if (e < N_EXPERTS)
        counts[e] = 0;
}

#elif STAGE == 2
KERNEL(gathermatmul_t1_dpas)(const __global int* indices, __global int* counts, __global int* pos, int rows) {
    const int gid = get_global_id(0);
    if (gid < rows)
        pos[gid] = atomic_inc(&counts[indices[gid]]);
}

#elif STAGE == 3
KERNEL(gathermatmul_t1_dpas)(const __global int* counts, __global int* offsets, __global int* subtiles, __global int* tiles, __global int* meta) {
    if (get_global_id(0) != 0)
        return;
    int ns = 0, nt = 0, off = 0;
    for (int e = 0; e < N_EXPERTS; ++e) {
        const int size = counts[e];
        offsets[e] = off;
        for (int r = 0; r < size; r += 8) {
            if ((r / 8) % SUBT == 0) {
                tiles[nt * 3] = e;
                tiles[nt * 3 + 1] = ns;
                tiles[nt * 3 + 2] = 0;
                ++nt;
            }
            tiles[(nt - 1) * 3 + 2] += 1;
            subtiles[ns * 2] = off + r;
            subtiles[ns * 2 + 1] = min(8, size - r);
            ++ns;
        }
        off += size;
    }
    meta[0] = ns;
    meta[1] = nt;
}

#elif STAGE == 4
KERNEL(gathermatmul_t1_dpas)(const __global int* indices,
                             const __global int* offsets,
                             const __global int* pos,
                             __global int* row_token,
                             __global int* row_slot,
                             int rows,
                             int top_k) {
    const int gid = get_global_id(0);
    if (gid >= rows)
        return;
    const int row = offsets[indices[gid]] + pos[gid];
    row_token[row] = gid / top_k;
    row_slot[row] = gid % top_k;
}

#elif STAGE == 5
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(gathermatmul_t1_dpas)(const __global half* X,
                             const __global int* row_token,
                             const __global int* row_slot,
                             __global float* XS,
                             __global int* XSUM,
                             int rows,
                             int n_tokens,
                             int n_act) {
    const uint sgid = get_global_id(0) / SIMD, lane = get_sub_group_local_id();
    const uint r = sgid / NG, g = sgid % NG;
    if (r >= (uint)rows)
        return;
    const __global half* a = X + SRC_ROW(r) * K_SIZE + g * G;
    float mx = 0.f;
    for (uint i = lane; i < G; i += SIMD)
        mx = fmax(mx, fabs((float)a[i]));
    mx = sub_group_reduce_max(mx);
    const float sc = mx > 0.f ? mx / 127.f : 1.f;
    int q = 0;
    for (uint i = lane; i < G; i += SIMD)
        q += QUANT_A(a[i], sc);
    q = sub_group_reduce_add(q);
    if (lane == 0) {
        XS[(size_t)r * NG + g] = sc;
        XSUM[(size_t)r * NG + g] = q;
    }
}

#elif STAGE == 6
KERNEL(gathermatmul_t1_dpas)(const __global half* X,
                             const __global int* row_token,
                             const __global int* row_slot,
                             const __global float* XS,
                             const __global int* subtiles,
                             const __global int* meta,
                             __global ushort* XD,
                             int n_tokens,
                             int n_act) {
    const size_t gid = get_global_id(0);
    const uint l = gid % 16, r = (gid / 16) % 8, c = (gid / 128) % NCH;
    const size_t unit = (gid / (128 * NCH)) % UNITS, s = gid / (128 * NCH * UNITS);
    if (s >= (size_t)meta[0])
        return;
    const int row0 = subtiles[s * 2], live = subtiles[s * 2 + 1];
    ushort out = 0;
    if ((int)r < live) {
        const size_t row = (size_t)row0 + r;
        const __global half* a = X + SRC_ROW(row) * K_SIZE;
        for (uint h = 0; h < 2; ++h) {
            const uint j = l / 2, t = (l % 2) * 2 + h;
            int v = 0;
#if WEIGHTS_U1
            const uint k = (c * 8 + j) / 4, i = (c * 8 + j) % 4;
            const uint b = i * 4 + t;
            const uint vi = unit * UNIT_VALUES + 64 * (b / 8) + 8 * (b % 8) + k;
            v = QUANT_A(a[vi], XS[row * NG + vi / G]);
#else
            int k, i;
            uint mask;
            FUNC_CALL(t1_op)(c, j, &k, &i, &mask);
            const uint b = i * 4 + t, p = 5 * (b % 13) + k;
            if (mask != 0u && p < 64) {
                const uint vi = unit * UNIT_VALUES + 64 * (b / 13) + p;
                v = QUANT_A(a[vi], XS[row * NG + vi / G]);
            }
#endif
            out |= (ushort)((v & 0xFF) << (8 * h));
        }
    }
    XD[gid] = out;
}

#elif STAGE == 7
REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(WG_SG * SIMD, 1, 1)))
KERNEL(gathermatmul_t1_dpas)(const __global ushort* XD,
                             const __global float* XS,
                             const __global int* XSUM,
                             const __global uchar* WALL,
                             const __global SCALE_T* SALL,
                             __global half* Y,
                             const __global int* row_token,
                             const __global int* row_slot,
                             const __global int* tiles,
                             const __global int* subtiles,
                             const __global int* meta,
                             int n_tokens) {
    const int tile = get_group_id(2);
    if (tile >= meta[1])
        return;
    const int expert = tiles[tile * 3], s0 = tiles[tile * 3 + 1], ns = tiles[tile * 3 + 2];
    const uint lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const uint n = get_group_id(1) * (WG_SG * SIMD) + sg * SIMD + lane;
    const uint nn = min(n, (uint)N_SIZE - 1);
    const __global uint* w = (const __global uint*)(WALL + ((size_t)expert * N_SIZE + nn) * ROW_BYTES);
    const __global SCALE_T* sc = SALL + (size_t)expert * NG * N_SIZE + nn;
    float acc[SUBT][8];
    unroll_for(int st = 0; st < SUBT; ++st) unroll_for(int r = 0; r < 8; ++r) acc[st][r] = 0.f;
    for (uint p = 0; p < UNITS; ++p) {
        uint B[NCH * 8];
#if WEIGHTS_U1
        uint wv[4];
        unroll_for(int i = 0; i < 4; ++i) wv[i] = w[p * 4 + i];
        unroll_for(int o = 0; o < 32; ++o) B[o] = (wv[o % 4] >> (7 - o / 4)) & 0x01010101u;
#else
        uint lo[13], hi[13], dg[5][13];
        unroll_for(int i = 0; i < 13; ++i) {
            const uint v = w[p * 13 + i];
            lo[i] = v & 0x00FF00FFu;
            hi[i] = (v >> 8) & 0x00FF00FFu;
        }
        unroll_for(int k = 0; k < 5; ++k) unroll_for(int i = 0; i < 13; ++i) {
            const uint tl = lo[i] * 3u, th = hi[i] * 3u;
            lo[i] = tl & 0x00FF00FFu;
            hi[i] = th & 0x00FF00FFu;
            dg[k][i] = ((tl >> 8) & 0x00030003u) | (((th >> 8) & 0x00030003u) << 8);
        }
        unroll_for(int c = 0; c < NCH; ++c) unroll_for(int j = 0; j < 8; ++j) {
            int k, i;
            uint mask;
            FUNC_CALL(t1_op)(c, j, &k, &i, &mask);
            B[c * 8 + j] = dg[k][i] & mask;
        }
#endif
        unroll_for(int st = 0; st < SUBT; ++st) {
            if (st < ns) {
                const int s = s0 + st;
                const __global ushort* xa = XD + (((size_t)s * UNITS + p) * NCH) * CHB;
                const int row0 = subtiles[s * 2];
                unroll_for(int grp = 0; grp < GPU_; ++grp) {
                    int8 ai = (int8)(0);
                    unroll_for(int c = grp * (NCH / GPU_); c < (grp + 1) * (NCH / GPU_); ++c) {
                        const short8 a = as_short8(intel_sub_group_block_read_us8(xa + c * CHB));
                        const int8 b = (int8)(B[c * 8 + 0], B[c * 8 + 1], B[c * 8 + 2], B[c * 8 + 3], B[c * 8 + 4], B[c * 8 + 5], B[c * 8 + 6],
                                              B[c * 8 + 7]);
                        ai = intel_sub_group_i8_i8_matrix_mad_k32(a, b, ai);
                    }
                    const uint g = p * GPU_ + grp;
                    const float ws = SCALE_VAL(sc[(size_t)g * N_SIZE]);
                    const int a8[8] = {ai.s0, ai.s1, ai.s2, ai.s3, ai.s4, ai.s5, ai.s6, ai.s7};
                    unroll_for(int r = 0; r < 8; ++r) {
                        // rows past 'live' read the (padded) neighbour rows; discarded at the store
                        const size_t rg = (size_t)(row0 + r) * NG + g;
                        acc[st][r] = fma(((float)a8[r] - ZP_F * (float)XSUM[rg]) * XS[rg], ws, acc[st][r]);
                    }
                }
            }
        }
    }
    if (n < N_SIZE) {
        unroll_for(int st = 0; st < SUBT; ++st) {
            if (st < ns) {
                const int s = s0 + st, row0 = subtiles[s * 2], live = subtiles[s * 2 + 1];
                unroll_for(int r = 0; r < 8; ++r) if (r < live) {
                    const int row = row0 + r;
                    Y[((size_t)row_slot[row] * n_tokens + row_token[row]) * N_SIZE + n] = (half)acc[st][r];
                }
            }
        }
    }
}
#endif

#undef SRC_ROW
#undef QUANT_A
#undef SIMD
#undef G
#undef NG
#undef UNIT_VALUES
#undef NCH
#undef ZP_F
#undef ROW_BYTES
#undef GPU_
#undef UNITS
#undef CHB
#undef WG_SG
#undef SCALE_T
#undef SCALE_VAL
