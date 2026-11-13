// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#if GATE_ACT_MUL
#include "moe_gate_activation.cl"
#endif

// Expert GEMMs of the fused MoE prefill for u3 (3-bit unsigned) weights, computed
// with dynamically quantized int8 activations:
//
//   C[r, n] = sum_g  a_scale[r, g] * w_scale[e, g, n] * sum_{k in g} A[r, k] * (W[e, n, k] - zp[e, g, n])
//
// where row r of the gathered activations belongs to expert e. The rows are
// sorted by expert, and offsets[e] is the exclusive end row of expert e. The
// activations and their scales come from moe_3gemm_dyn_quan.cl. The zero point
// is subtracted while the weights are decoded: for a zero point in the u3 range
// [0, 7], W - zp lies in [-7, 7], still an int4 DPAS operand. With GATE_ACT_MUL
// (the gate GEMM), the output is act(C) * up, with up the output of the up GEMM.
//
// Weights are the plain [E, N, K] u3 tensor: each output channel is a
// LSB-first bit stream of K values, so a run of 32 values (a granule) is exactly
// 3 uints and is the DPAS b operand of one K step once unpacked.
//
// The second grid dimension walks a host-built list of {expert, first row}
// tiles of SG_M * TILE_M rows. The SG_M subgroups of a work-group share one
// decoded weight tile through SLM and each multiply it with TILE_M rows through
// DPAS.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define SIMD             16
#define K_CHUNK          32                        // u3 values per 3-uint granule, and the DPAS K step
#define GRANULE_UINTS    3
#define CHUNKS_PER_GROUP (GROUP_SIZE / K_CHUNK)
#define CHUNKS_K         (K_SIZE / K_CHUNK)
#define GROUPS_K         (K_SIZE / GROUP_SIZE)
#define W_GROUPS_K       (K_SIZE / WEI_GROUP_SIZE)

// Eight consecutive u3 values (the low 24 bits of f; higher bits are ignored)
// spread into the eight nibbles of an int: into 12-bit halves at bits 0 and 16,
// then 6-bit quarters at each byte, then 3-bit values at each nibble. Each step
// keeps the low part in place and takes the rest from the shifted copy (one bfn);
// the bits the steps leave over are cleared at the end.
inline int FUNC(u3_spread8_nib)(uint f) {
    uint u = bitselect(f << 4, f, 0x00000FFFu);
    u = bitselect(u << 2, u, 0x003F003Fu);
    u = bitselect(u << 1, u, 0x07070707u);
    return as_int(u & 0x77777777u);
}

// One granule as the b operand of the int8 x int4 DPAS: value k at nibble k % 8
// of component k / 8.
inline int4 FUNC(u3_to_dpas_b4)(uint w0, uint w1, uint w2) {
    return (int4)(FUNC_CALL(u3_spread8_nib)(w0),
                  FUNC_CALL(u3_spread8_nib)((w0 >> 24) | (w1 << 8)),
                  FUNC_CALL(u3_spread8_nib)((w1 >> 16) | (w2 << 16)),
                  FUNC_CALL(u3_spread8_nib)(w2 >> 8));
}

// Subtracts the zero point z in [0, 7] from each of the 32 nibbles of b (values in [0, 7]): with bit 3 set first, a
// nibble holds v + 8 in [8, 15], and v + 8 - z in [1, 15] borrows nothing from its neighbor; clearing bit 3 again
// subtracts 8 modulo 16, which leaves v - z as a signed int4.
inline int4 FUNC(sub_zp_b4)(int4 b, uint z) {
    return as_int4(((as_uint4(b) | 0x88888888u) - z * 0x11111111u) ^ 0x88888888u);
}

// Scale and zero point of expert e, output channel n, activation group g. Both
// are [E, W_GROUPS_K, N] in memory; a scalar zero point is shared by all experts.
#define WEI_SCALE(e, n, g) \
    ((float)scale[((e) * W_GROUPS_K + (g) * GROUP_SIZE / WEI_GROUP_SIZE) * N_SIZE + (n)])
#if ZP_SCALAR
#   define WEI_ZP(e, n, g) ((int)zp[0])
#else
#   define WEI_ZP(e, n, g) \
        ((int)zp[((e) * W_GROUPS_K + (g) * GROUP_SIZE / WEI_GROUP_SIZE) * N_SIZE + (n)])
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(SIMD, SG_M, 1)))
KERNEL(moe_int3_gemm)(
    const __global char* quantized_input,   // [rows, K]
    const __global half* a_scales,          // [rows, GROUPS_K]
    const __global uint* weights,           // [E, N, K] u3
    const __global half* scale,             // [E, W_GROUPS_K, N]
    const __global ZP_TYPE* zp,             // [E, W_GROUPS_K, N] or [1]
    const __global int* offsets,            // [E] exclusive end row of each expert
    const __global int2* tiles,             // [tiles] {expert, first row of the tile within the expert}
#if GATE_ACT_MUL
    const __global half* up,                // [rows, N]
#endif
    __global half* output)                  // [rows, N]
{
    const uint lane = get_sub_group_local_id();
    const uint sg   = (uint)get_local_id(1);

    // A work-group takes one {expert, first row} entry of the host's tile list, so
    // only row tiles that hold rows are launched. A subgroup computes TILE_M (16) rows x
    // NB 16-column blocks, so each activation load feeds NB DPAS. The activations
    // of a quantization group arrive through 2D block reads anchored at the
    // expert's first row: their layout is exactly the DPAS a operand, and rows past
    // the expert read as zero. The weights minus the zero point are staged as signed
    // 4-bit values for the int8 x int4 DPAS and shared by the SG_M
    // subgroups through a triple-buffered SLM stage behind a split barrier: the
    // next group is staged and the barrier entered before the current group is
    // computed, and only waited on afterwards. A buffer is rewritten two groups
    // after it was read, and passing the wait of the group in between proves every
    // subgroup is done with it. Subgroups whose rows lie past the expert's end
    // still stage their share of the weights, but skip the DPAS and the rescale.
#define NH              (CHUNKS_PER_GROUP / 2)
#define GRAN_PER_GROUP  (CHUNKS_PER_GROUP * NB)
#define GRAN_PER_SG     (GRAN_PER_GROUP / SG_M)
#if TILE_M != 16 || (GRAN_PER_GROUP % SG_M) != 0 || (CHUNKS_PER_GROUP % 2) != 0
#   error "moe_3gemm_int3_gemm.cl - invalid tile configuration"
#endif
    const int2 tile = tiles[get_group_id(1)];
    const uint expert = (uint)tile.x;
    const int row_begin = expert == 0 ? 0 : offsets[expert - 1];
    const int rows = offsets[expert] - row_begin;
    const uint m0 = (uint)tile.y + sg * TILE_M;
    const uint n0 = (uint)get_group_id(0) * NB * SIMD;
    const bool has_rows = (int)m0 < rows;

    const __global char* A = quantized_input + (size_t)row_begin * K_SIZE;
    // Activation scales of row m0 + lane
    const size_t V = (size_t)(row_begin + min((int)(m0 + lane), rows - 1)) * GROUPS_K;

    float acc_f[NB][TILE_M];
    unroll_for (uint j = 0; j < NB; ++j)
        unroll_for (uint t = 0; t < TILE_M; ++t)
            acc_f[j][t] = 0.0f;

    __local int4 wshare[3][GRAN_PER_GROUP][SIMD];
    uint3 raw[GRAN_PER_SG];
    uint raw_zp[GRAN_PER_SG];

    // Granule i of this subgroup in group g: column block gidx / CHUNKS_PER_GROUP,
    // chunk gidx % CHUNKS_PER_GROUP; staged at SLM slot chunk * NB + block.
#define GRAN_GIDX(i) (sg * GRAN_PER_SG + (i))
#define GRAN_SLOT(i) ((GRAN_GIDX(i) % CHUNKS_PER_GROUP) * NB + GRAN_GIDX(i) / CHUNKS_PER_GROUP)
#define LOAD_RAW(g_)                                                                \
    unroll_for (uint i = 0; i < GRAN_PER_SG; ++i) {                                 \
        const uint col = n0 + (GRAN_GIDX(i) / CHUNKS_PER_GROUP) * SIMD + lane;      \
        const __global uint* wp = weights + ((size_t)expert * N_SIZE + col) * CHUNKS_K * GRANULE_UINTS; \
        raw[i] = vload3((g_) * CHUNKS_PER_GROUP + GRAN_GIDX(i) % CHUNKS_PER_GROUP, wp); \
        raw_zp[i] = (uint)WEI_ZP(expert, col, g_);                                  \
    }
#define STAGE_RAW(b_)                                                               \
    unroll_for (uint i = 0; i < GRAN_PER_SG; ++i)                                   \
        wshare[b_][GRAN_SLOT(i)][lane] =                                            \
            FUNC_CALL(sub_zp_b4)(FUNC_CALL(u3_to_dpas_b4)(raw[i].s0, raw[i].s1, raw[i].s2), raw_zp[i]);

    LOAD_RAW(0)
    STAGE_RAW(0)
    barrier(CLK_LOCAL_MEM_FENCE);
    if (1 < GROUPS_K) {
        LOAD_RAW(1)
    }

    for (uint g = 0; g < GROUPS_K; ++g) {
        const uint buf = g % 3u;
        if (g + 1 < GROUPS_K) {
            STAGE_RAW((g + 1) % 3u)
            intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
            if (g + 2 < GROUPS_K) {
                LOAD_RAW(g + 2)
            }
        }

        if (has_rows) {
            float bs[NB];
            unroll_for (uint j = 0; j < NB; ++j)
                bs[j] = WEI_SCALE(expert, n0 + j * SIMD + lane, g);
            const float a_scale = (float)a_scales[V + g];

            ushort a_all[NH][32];
            unroll_for (uint h = 0; h < NH; ++h)
                intel_sub_group_2d_block_read_8b_16r32x2c((__global void*)A, K_SIZE, rows, K_SIZE,
                                                          (int2)(g * GROUP_SIZE + h * 2 * K_CHUNK, m0),
                                                          a_all[h]);

            // Rows 0-7 and 8-15 of the tile
            int8 acc[2][NB];
            unroll_for (uint mb = 0; mb < 2; ++mb)
                unroll_for (uint j = 0; j < NB; ++j)
                    acc[mb][j] = (int8)(0);

            unroll_for (uint h = 0; h < NH; ++h) {
                unroll_for (uint c = 0; c < 2; ++c) {
                    const uint cc = h * 2 + c;
                    int4 w[NB];
                    unroll_for (uint j = 0; j < NB; ++j)
                        w[j] = wshare[buf][cc * NB + j][lane];
                    unroll_for (uint mb = 0; mb < 2; ++mb) {
                        short8 a;
                        unroll_for (uint t = 0; t < 8; ++t)
                            a[t] = as_short(a_all[h][c * 16 + mb * 8 + t]);
                        unroll_for (uint j = 0; j < NB; ++j)
                            acc[mb][j] = intel_sub_group_i8_i4_matrix_mad_k32(a, w[j], acc[mb][j]);
                    }
                }
            }

            unroll_for (uint mb = 0; mb < 2; ++mb) {
                unroll_for (uint t = 0; t < 8; ++t) {
                    const uint r = mb * 8 + t;
                    const float as = sub_group_broadcast(a_scale, r);
                    unroll_for (uint j = 0; j < NB; ++j)
                        acc_f[j][r] = fma((float)acc[mb][j][t] * as, bs[j], acc_f[j][r]);
                }
            }
        }

        if (g + 1 < GROUPS_K)
            intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
    }
#undef GRAN_GIDX
#undef GRAN_SLOT
#undef LOAD_RAW
#undef STAGE_RAW

    unroll_for (uint t = 0; t < TILE_M; ++t) {
        if ((int)(m0 + t) < rows) {
            unroll_for (uint j = 0; j < NB; ++j) {
                const size_t idx = (size_t)(row_begin + m0 + t) * N_SIZE + n0 + j * SIMD + lane;
#if GATE_ACT_MUL
                output[idx] = (half)(MOE_GATE_ACT(acc_f[j][t]) * (float)up[idx]);
#else
                output[idx] = (half)acc_f[j][t];
#endif
            }
        }
    }
#undef NH
#undef GRAN_PER_GROUP
#undef GRAN_PER_SG
}

#undef SIMD
#undef K_CHUNK
#undef GRANULE_UINTS
#undef CHUNKS_PER_GROUP
#undef CHUNKS_K
#undef GROUPS_K
#undef W_GROUPS_K
