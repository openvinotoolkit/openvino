// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Expert GEMMs of the fused MoE prefill for u3 (3-bit unsigned) weights, computed
// with dynamically quantized int8 activations:
//
//   C[r, n] = sum_g  a_scale[r, g] * w_scale[e, g, n] *
//                    ( sum_{k in g} A[r, k] * W[e, n, k]  -  zp[e, g, n] * sum_{k in g} A[r, k] )
//
// where row r of the gathered activations belongs to expert e. The rows are
// sorted by expert, and offsets[e] is the exclusive end row of expert e. The
// activations, their scales and their sums come from moe_3gemm_dyn_quan.cl.
//
// Weights are the plain [E, N, K] u3 tensor: each output channel is a
// LSB-first bit stream of K values, so a run of 32 values (a granule) is exactly
// 3 uints and is the DPAS b operand of one K step once unpacked.
//
// GEMM has two dispatch modes:
//  - ROW_MODE (few rows per expert): one work-group per gathered row, which
//    finds its expert in offsets. SG_K subgroups split K and reduce through SLM.
//  - expert mode: the second grid dimension walks a host-built list of
//    {expert, first row} tiles of SG_M * TILE_M rows. The SG_M subgroups of a
//    work-group share one decoded weight tile through SLM and each multiply it
//    with TILE_M rows through DPAS.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#if !ROW_MODE
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#endif

#define SIMD             16
#define K_CHUNK          32                        // u3 values per 3-uint granule, and the DPAS K step
#define GRANULE_UINTS    3
#define CHUNKS_PER_GROUP (GROUP_SIZE / K_CHUNK)
#define CHUNKS_K         (K_SIZE / K_CHUNK)
#define GROUPS_K         (K_SIZE / GROUP_SIZE)
#define W_GROUPS_K       (K_SIZE / WEI_GROUP_SIZE)

// Bit-stream extraction of value i (a literal) from the three granule words.
// Both branches fold away at compile time; only i == 10 and i == 21 straddle.
#define U3_WORD(w0, w1, w2, idx) ((idx) == 0 ? (w0) : ((idx) == 1 ? (w1) : (w2)))
#define U3_BIT(i) (3u * (i))
#define U3_IDX(i) (U3_BIT(i) >> 5)
#define U3_OFF(i) (U3_BIT(i) & 31u)
#define U3_AT(w0, w1, w2, i)                                                        \
    ((U3_OFF(i) <= 29u)                                                             \
         ? ((U3_WORD(w0, w1, w2, U3_IDX(i)) >> U3_OFF(i)) & 7u)                     \
         : (((U3_WORD(w0, w1, w2, U3_IDX(i)) >> U3_OFF(i)) |                        \
             (U3_WORD(w0, w1, w2, U3_IDX(i) + 1u) << (32u - U3_OFF(i)))) & 7u))

// Four consecutive weights as a char4, without the zero point: it is applied
// once per group through the activation sum.
#define U3_CHAR4(w0, w1, w2, i)                                                     \
    (char4)((char)U3_AT(w0, w1, w2, (i) + 0), (char)U3_AT(w0, w1, w2, (i) + 1),      \
            (char)U3_AT(w0, w1, w2, (i) + 2), (char)U3_AT(w0, w1, w2, (i) + 3))

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

// Scale and zero point of expert e, output channel n, activation group g. Both
// are [E, W_GROUPS_K, N] in memory; a scalar zero point is shared by all experts.
#define WEI_SCALE(e, n, g) \
    ((float)scale[((e) * W_GROUPS_K + (g) * GROUP_SIZE / WEI_GROUP_SIZE) * N_SIZE + (n)])
#if ZP_SCALAR
#   define WEI_ZP(e, n, g) ((float)zp[0])
#else
#   define WEI_ZP(e, n, g) \
        ((float)zp[((e) * W_GROUPS_K + (g) * GROUP_SIZE / WEI_GROUP_SIZE) * N_SIZE + (n)])
#endif

inline int FUNC(mad4)(char4 a, char4 b, int acc) {
    acc += (int)a.x * (int)b.x;
    acc += (int)a.y * (int)b.y;
    acc += (int)a.z * (int)b.z;
    acc += (int)a.w * (int)b.w;
    return acc;
}

#if ROW_MODE
#define SG_COUNT SG_K
#else
#define SG_COUNT SG_M
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(SIMD, SG_COUNT, 1)))
KERNEL(moe_int3_gemm)(
    const __global char* quantized_input,   // [rows, K]
    const __global half* a_scales,          // [rows, GROUPS_K]
    const __global int* a_sums,             // [rows, GROUPS_K]
    const __global uint* weights,           // [E, N, K] u3
    const __global half* scale,             // [E, W_GROUPS_K, N]
    const __global ZP_TYPE* zp,             // [E, W_GROUPS_K, N] or [1]
    const __global int* offsets,            // [E] exclusive end row of each expert
#if !ROW_MODE
    const __global int2* tiles,             // [tiles] {expert, first row of the tile within the expert}
#endif
    __global half* output,                  // [rows, N]
    const int num_experts)
{
    const uint lane = get_sub_group_local_id();
    const uint sg   = (uint)get_local_id(1);
    const uint n    = (uint)get_group_id(0) * SIMD + lane;

#if ROW_MODE
    // Expert of this row: the first one whose end offset lies past it.
    const int row = (int)get_group_id(2);
    int lo = 0;
    int hi = num_experts - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] > row)
            hi = mid;
        else
            lo = mid + 1;
    }
    const uint expert = (uint)lo;

    const __global uint* B = weights + ((size_t)expert * N_SIZE + n) * CHUNKS_K * GRANULE_UINTS;
    const __global uint* A = (const __global uint*)(quantized_input + (size_t)row * K_SIZE);
    const __global half* AS = a_scales + (size_t)row * GROUPS_K;
    const __global int* AZ = a_sums + (size_t)row * GROUPS_K;

    float out = 0.0f;
    for (uint g = sg; g < GROUPS_K; g += SG_K) {
        int acc = 0;
        unroll_for (uint cc = 0; cc < CHUNKS_PER_GROUP; ++cc) {
            const uint chunk = g * CHUNKS_PER_GROUP + cc;
            const uint3 w = vload3(chunk, B);
            const uint4 a0 = vload4(0, A + chunk * 8);
            const uint4 a1 = vload4(1, A + chunk * 8);
            acc = FUNC_CALL(mad4)(as_char4(a0.s0), U3_CHAR4(w.s0, w.s1, w.s2, 0), acc);
            acc = FUNC_CALL(mad4)(as_char4(a0.s1), U3_CHAR4(w.s0, w.s1, w.s2, 4), acc);
            acc = FUNC_CALL(mad4)(as_char4(a0.s2), U3_CHAR4(w.s0, w.s1, w.s2, 8), acc);
            acc = FUNC_CALL(mad4)(as_char4(a0.s3), U3_CHAR4(w.s0, w.s1, w.s2, 12), acc);
            acc = FUNC_CALL(mad4)(as_char4(a1.s0), U3_CHAR4(w.s0, w.s1, w.s2, 16), acc);
            acc = FUNC_CALL(mad4)(as_char4(a1.s1), U3_CHAR4(w.s0, w.s1, w.s2, 20), acc);
            acc = FUNC_CALL(mad4)(as_char4(a1.s2), U3_CHAR4(w.s0, w.s1, w.s2, 24), acc);
            acc = FUNC_CALL(mad4)(as_char4(a1.s3), U3_CHAR4(w.s0, w.s1, w.s2, 28), acc);
        }
        const float part = (float)acc - WEI_ZP(expert, n, g) * (float)AZ[g];
        out += part * (float)AS[g] * WEI_SCALE(expert, n, g);
    }

#if SG_K > 1
    __local float partial[SG_K][SIMD];
    partial[sg][lane] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0)
        return;
    unroll_for (uint j = 1; j < SG_K; ++j)
        out += partial[j][lane];
#endif
    output[(size_t)row * N_SIZE + n] = (half)out;

#else  // expert mode
    // A work-group takes one {expert, first row} entry of the host's tile list, so
    // only row tiles that hold rows are launched. A subgroup computes TILE_M (16) rows x
    // NB 16-column blocks, so each activation load feeds NB DPAS. The activations
    // of a quantization group arrive through 2D block reads anchored at the
    // expert's first row: their layout is exactly the DPAS a operand, and rows past
    // the expert read as zero. The weights are staged as 4-bit values for the int8
    // x int4 DPAS (u3 values are valid signed int4) and shared by the SG_M
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
#   error "moe_3gemm_int3_gemm.cl - invalid expert mode configuration"
#endif
    const int2 tile = tiles[get_group_id(1)];
    const uint expert = (uint)tile.x;
    const int row_begin = expert == 0 ? 0 : offsets[expert - 1];
    const int rows = offsets[expert] - row_begin;
    const uint m0 = (uint)tile.y + sg * TILE_M;
    const uint n0 = (uint)get_group_id(0) * NB * SIMD;
    const bool has_rows = (int)m0 < rows;

    const __global char* A = quantized_input + (size_t)row_begin * K_SIZE;
    // Activation scales and sums of row m0 + lane
    const size_t V = (size_t)(row_begin + min((int)(m0 + lane), rows - 1)) * GROUPS_K;
#if ZP_SCALAR
    const float zp_scalar = (float)zp[0];
#endif

    float acc_f[NB][TILE_M];
    unroll_for (uint j = 0; j < NB; ++j)
        unroll_for (uint t = 0; t < TILE_M; ++t)
            acc_f[j][t] = 0.0f;

    __local int4 wshare[3][GRAN_PER_GROUP][SIMD];
    uint3 raw[GRAN_PER_SG];

    // Granule i of this subgroup in group g: column block gidx / CHUNKS_PER_GROUP,
    // chunk gidx % CHUNKS_PER_GROUP; staged at SLM slot chunk * NB + block.
#define GRAN_GIDX(i) (sg * GRAN_PER_SG + (i))
#define GRAN_SLOT(i) ((GRAN_GIDX(i) % CHUNKS_PER_GROUP) * NB + GRAN_GIDX(i) / CHUNKS_PER_GROUP)
#define LOAD_RAW(g_)                                                                \
    unroll_for (uint i = 0; i < GRAN_PER_SG; ++i) {                                 \
        const uint col = n0 + (GRAN_GIDX(i) / CHUNKS_PER_GROUP) * SIMD + lane;      \
        const __global uint* wp = weights + ((size_t)expert * N_SIZE + col) * CHUNKS_K * GRANULE_UINTS; \
        raw[i] = vload3((g_) * CHUNKS_PER_GROUP + GRAN_GIDX(i) % CHUNKS_PER_GROUP, wp); \
    }
#define STAGE_RAW(b_)                                                               \
    unroll_for (uint i = 0; i < GRAN_PER_SG; ++i)                                   \
        wshare[b_][GRAN_SLOT(i)][lane] = FUNC_CALL(u3_to_dpas_b4)(raw[i].s0, raw[i].s1, raw[i].s2);

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
#if !ZP_SCALAR
            float bzp[NB];
#endif
            unroll_for (uint j = 0; j < NB; ++j) {
                const uint col = n0 + j * SIMD + lane;
                bs[j] = WEI_SCALE(expert, col, g);
#if !ZP_SCALAR
                bzp[j] = WEI_ZP(expert, col, g);
#endif
            }
            const float a_scale = (float)a_scales[V + g];
            const float a_sum = (float)a_sums[V + g];

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
                    const float as_sum = as * sub_group_broadcast(a_sum, r);
                    unroll_for (uint j = 0; j < NB; ++j) {
#if ZP_SCALAR
                        const float part = fma((float)acc[mb][j][t], as, -zp_scalar * as_sum);
#else
                        const float part = fma((float)acc[mb][j][t], as, -bzp[j] * as_sum);
#endif
                        acc_f[j][r] = fma(part, bs[j], acc_f[j][r]);
                    }
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
            unroll_for (uint j = 0; j < NB; ++j)
                output[(size_t)(row_begin + m0 + t) * N_SIZE + n0 + j * SIMD + lane] = (half)acc_f[j][t];
        }
    }
#undef NH
#undef GRAN_PER_GROUP
#undef GRAN_PER_SG
#endif  // ROW_MODE
}

#undef SG_COUNT
#undef SIMD
#undef K_CHUNK
#undef GRANULE_UINTS
#undef CHUNKS_PER_GROUP
#undef CHUNKS_K
#undef GROUPS_K
#undef W_GROUPS_K
