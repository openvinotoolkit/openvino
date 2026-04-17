// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimised GEMV for the LLM token-generation phase (M=1).
//
// Branch A (IS_WEIGHT_INT4==0): f16/f16 pure GEMM, N-parallel dispatch.
// Branch B (IS_WEIGHT_INT4==1): K-parallel sub-group approach for W4A16/W4A8.
//   Each sub-group of SG_SIZE=16 lanes cooperates on K-reduction for TILE_N=2
//   output channels using intel_sub_group_block_read_{us8,uc4} for coalesced
//   memory access, then sub_group_reduce_add for cross-lane reduction.
//   Dispatch: global=[WG_SIZE, B, N/TILE_N], local=[WG_SIZE, 1, 1]

#include "include/batch_headers/common.cl"

// Horizontal sum of a float8.
#define HSUM8(v) ((v).s0 + (v).s1 + (v).s2 + (v).s3 + \
                  (v).s4 + (v).s5 + (v).s6 + (v).s7)

// ============================================================
// Branch A: f16/f16 pure GEMM -- N-parallel (each lane = one N)
// ============================================================
#if !defined(IS_WEIGHT_INT4) || (IS_WEIGHT_INT4 == 0)

#define INPUT_VEC_TYPE   MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define WEIGHT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT1_TYPE, VEC_SIZE)
#define ACC_VEC_TYPE     MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE)
#define INPUT_VLOAD      CAT(vload, VEC_SIZE)
#define TO_ACC_VEC8(x)   convert_float8(x)

KERNEL(gemm_generate_opt)(
    const __global INPUT0_TYPE*  A,   // activations [B, K]
    const __global INPUT1_TYPE*  B,   // weights     [B, N, K]
          __global OUTPUT_TYPE*  C    // output      [B, N]
)
{
    const int n = (int)get_global_id(0);
    const int b = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    const int a_base = b * K_SIZE;
    const int w_base = b * N_SIZE * K_SIZE + n * K_SIZE;

    ACC_VEC_TYPE acc = (ACC_VEC_TYPE)(ACCUMULATOR_VAL_ZERO);

    const int k_iters = K_SIZE / VEC_SIZE;
    for (int ki = 0; ki < k_iters; ++ki) {
        INPUT_VEC_TYPE  a_vec = INPUT_VLOAD(ki, A + a_base);
        WEIGHT_VEC_TYPE w_vec = INPUT_VLOAD(ki, B + w_base);
        acc = mad(TO_ACC_VEC8(a_vec), TO_ACC_VEC8(w_vec), acc);
    }

    ACCUMULATOR_TYPE result = (ACCUMULATOR_TYPE)(HSUM8(acc));
    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(result);
}

// ============================================================
// Branch B: f16-activation + int4-weight WOQ GEMV
// ============================================================
#else  // IS_WEIGHT_INT4 == 1

// Unpack nibbles from a uchar4 into 8 floats (low nibble = lower K index).
#if WEIGHT_IS_SIGNED
#define SIGN_EXT4(x) ((x) >= 8 ? (int)(x) - 16 : (int)(x))
#define UNPACK8(b4, w)                  \
    (w).s0 = (float)(SIGN_EXT4((b4).s0 & 0xF));  \
    (w).s1 = (float)(SIGN_EXT4((b4).s0 >> 4));   \
    (w).s2 = (float)(SIGN_EXT4((b4).s1 & 0xF));  \
    (w).s3 = (float)(SIGN_EXT4((b4).s1 >> 4));   \
    (w).s4 = (float)(SIGN_EXT4((b4).s2 & 0xF));  \
    (w).s5 = (float)(SIGN_EXT4((b4).s2 >> 4));   \
    (w).s6 = (float)(SIGN_EXT4((b4).s3 & 0xF));  \
    (w).s7 = (float)(SIGN_EXT4((b4).s3 >> 4))
#else
#define UNPACK8(b4, w)                       \
    (w).s0 = (float)((b4).s0 & 0xF);        \
    (w).s1 = (float)((b4).s0 >> 4);         \
    (w).s2 = (float)((b4).s1 & 0xF);        \
    (w).s3 = (float)((b4).s1 >> 4);         \
    (w).s4 = (float)((b4).s2 & 0xF);        \
    (w).s5 = (float)((b4).s2 >> 4);         \
    (w).s6 = (float)((b4).s3 & 0xF);        \
    (w).s7 = (float)((b4).s3 >> 4)
#endif  // WEIGHT_IS_SIGNED

// Weight layout: raw [N, K/2] packed bytes (u4/i4).
#define K_HALF_SIZE (K_SIZE / 2)

// -----------------------------------------------------------------------
// ZP dequant helper macro.
// Scale and ZP tensors use fbyx memory format.
//   scale(n, gk) = Scale[gk * N_SIZE + n]
//   zp_u8(n, gk) = ZP[gk * N_SIZE + n]
//   zp_u4(n, gk): byte = ZP[(gk/2) * N_SIZE + n], nibble = gk%2
// -----------------------------------------------------------------------
#if HAS_ZP
#if ZP_IS_U8
#define LOAD_ZP(gk, n, ZP, zp_var)  (zp_var) = (float)ZP[(gk) * N_SIZE + (n)]
#else
#define LOAD_ZP(gk, n, ZP, zp_var)                                         \
    do {                                                                    \
        uchar zp_byte_ = ZP[((gk) / 2) * N_SIZE + (n)];                   \
        (zp_var) = ((gk) % 2 == 0) ? (float)(zp_byte_ & 0xF)              \
                                    : (float)(zp_byte_ >> 4);              \
    } while (0)
#endif  // ZP_IS_U8
#else
#define LOAD_ZP(gk, n, ZP, zp_var)  (zp_var) = 0.0f
#endif  // HAS_ZP

// Default WG_SIZE fallback (should be provided by JIT).
#ifndef WG_SIZE
#define WG_SIZE 256
#endif

// Number of output channels computed by each work-item.
#ifndef TILE_N
#define TILE_N 1
#endif

// Unpack 8 u4 values from a single uint (4 bytes) using bitwise shifts.
// More efficient than uchar4 approach: single 4-byte read + shifts.
#if WEIGHT_IS_SIGNED
#define UNPACK8_UINT(packed, w)                                     \
    (w).s0 = (float)((int)((packed)       & 0xFu) - (((packed)       & 0x8u) ? 16 : 0)); \
    (w).s1 = (float)((int)(((packed) >> 4) & 0xFu) - ((((packed) >> 4) & 0x8u) ? 16 : 0)); \
    (w).s2 = (float)((int)(((packed) >> 8) & 0xFu) - ((((packed) >> 8) & 0x8u) ? 16 : 0)); \
    (w).s3 = (float)((int)(((packed) >> 12) & 0xFu) - ((((packed) >> 12) & 0x8u) ? 16 : 0)); \
    (w).s4 = (float)((int)(((packed) >> 16) & 0xFu) - ((((packed) >> 16) & 0x8u) ? 16 : 0)); \
    (w).s5 = (float)((int)(((packed) >> 20) & 0xFu) - ((((packed) >> 20) & 0x8u) ? 16 : 0)); \
    (w).s6 = (float)((int)(((packed) >> 24) & 0xFu) - ((((packed) >> 24) & 0x8u) ? 16 : 0)); \
    (w).s7 = (float)((int)(((packed) >> 28) & 0xFu) - ((((packed) >> 28) & 0x8u) ? 16 : 0))
#else
#define UNPACK8_UINT(packed, w)                       \
    (w).s0 = (float)((packed)       & 0xFu);         \
    (w).s1 = (float)(((packed) >> 4) & 0xFu);        \
    (w).s2 = (float)(((packed) >> 8) & 0xFu);        \
    (w).s3 = (float)(((packed) >> 12) & 0xFu);       \
    (w).s4 = (float)(((packed) >> 16) & 0xFu);       \
    (w).s5 = (float)(((packed) >> 20) & 0xFu);       \
    (w).s6 = (float)(((packed) >> 24) & 0xFu);       \
    (w).s7 = (float)(((packed) >> 28) & 0xFu)
#endif

// Unpack a uchar8 (16 nibbles) into two float8 vectors for wider processing.
#if WEIGHT_IS_SIGNED
#define UNPACK16(b8, w0, w1)                                  \
    UNPACK8((uchar4)((b8).s0,(b8).s1,(b8).s2,(b8).s3), w0);  \
    UNPACK8((uchar4)((b8).s4,(b8).s5,(b8).s6,(b8).s7), w1)
#else
#define UNPACK16(b8, w0, w1)                                  \
    UNPACK8((uchar4)((b8).s0,(b8).s1,(b8).s2,(b8).s3), w0);  \
    UNPACK8((uchar4)((b8).s4,(b8).s5,(b8).s6,(b8).s7), w1)
#endif

// 4-bit dequant helpers for K-parallel sub-group reads.
#if WEIGHT_IS_SIGNED
#define DEQUANT_4BIT_LO(v) convert_half((char)(((v) & 0x08) ? ((v) | 0xF0) : ((v) & 0x0F)))
#define DEQUANT_4BIT_HI(v) convert_half((char)((((v) >> 4) & 0x08) ? (((v) >> 4) | 0xF0) : (((v) >> 4) & 0x0F)))
#else
#define DEQUANT_4BIT_LO(v) convert_half((v) & 0x0F)
#define DEQUANT_4BIT_HI(v) convert_half((v) >> 4)
#endif

// Iteration chunk size for K-parallel path — must divide GROUP_SIZE.
#ifndef FAKE_GROUP_SIZE
#define FAKE_GROUP_SIZE 128
#endif

// ============================================================
// Branch B1: f16-activation + int4-weight WOQ GEMV  (W4A16)
// SLM-based sub-group cooperative approach.
// ============================================================
#if !defined(IS_ACT_INT8) || (IS_ACT_INT8 == 0)

// -----------------------------------------------------------------------
// SLM-based sub-group cooperative GEMV (optimized per dev_gemv_opt tips).
//
// Design rationale (memory-bound):
//   Cache activation vector in SLM (reused across all output rows).
//   Stream weights from global memory via intel_sub_group_block_read.
//   Each subgroup processes 2 output channels (N-block tiling, n+=2).
//   Interleave activation in SLM to align with u4 lo/hi nibble unpacking.
//   Pre-compute xg_sum for asymmetric quantization zero-point optimization.
//   Use half-precision FMA for u4 dequant path, float accumulator for stability.
//   Single sub_group_reduce_add at the end, lane 0 writes result.
//
// Dispatch:  global = [N_BLOCK_WG * num_wg, B, 1],  local = [WG_SIZE, 1, 1]
//   where N_BLOCK_WG = (WG_SIZE / SG_SIZE) * 2 = num_subgroups * 2
//   Work-group index in dim0 = get_group_id(0) selects N-block.
//   Each subgroup within the work-group handles 2 output channels.
// -----------------------------------------------------------------------

#define unroll_for __attribute__((opencl_unroll_hint)) for

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(gemm_generate_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE*  A,         // f16 activation [B, K]
    const __global uchar*        W,         // u4/i4 packed weight [N, K/2]
    const __global INPUT2_TYPE*  Scale,     // f16 scale, fbyx: [NUM_GROUPS, N]
#if HAS_ZP
    const __global uchar*        ZP,        // ZP fbyx
#endif
    __global OUTPUT_TYPE*        C          // f16 output [B, N]
)
{
    const int id_sg    = get_sub_group_id();
    const int num_sg   = get_num_sub_groups();
    const int id_local = get_sub_group_local_id();
    const int b        = (int)get_global_id(1);

    // N-block assignment: each work-group handles num_sg * 2 output channels.
    const int n_block_wg = num_sg * TILE_N;
    const int wg_idx     = get_group_id(0);
    const int n_wg_start = wg_idx * n_block_wg;

    // Base activation pointer for this batch.
    const __global half* A_row = A + b * K_SIZE;

    // ---- SLM: activation cache + xg_sum for ZP optimization ----
    __local half x_slm[K_SIZE];
#if HAS_ZP
    __local float xg_sum[K_SIZE / FAKE_GROUP_SIZE];
#endif

    // ---- Phase 1: Cooperative SLM loading with interleaving for u4 ----
    // All subgroups collaborate: each loads chunk[id_sg], stride by num_sg.
    {
        const __global half* px = A_row + id_sg * FAKE_GROUP_SIZE;
        __local half* px2 = x_slm + id_sg * FAKE_GROUP_SIZE;
        unroll_for(int i = id_sg; i < K_SIZE / FAKE_GROUP_SIZE;
                   i += num_sg, px += num_sg * FAKE_GROUP_SIZE, px2 += num_sg * FAKE_GROUP_SIZE) {
#if HAS_ZP
            float x_group_sum = 0;
#endif
            // Interleave: even elements in first half, odd in second half.
            // This aligns with DEQUANT_4BIT_LO (first half) / DEQUANT_4BIT_HI (second half).
            unroll_for(int j = id_local; j < FAKE_GROUP_SIZE / 2; j += SG_SIZE) {
                half even = px[2 * j + 0];
                half odd  = px[2 * j + 1];
                px2[j]                       = even;
                px2[j + FAKE_GROUP_SIZE / 2] = odd;
#if HAS_ZP
                x_group_sum += even + odd;
#endif
            }
#if HAS_ZP
            x_group_sum = sub_group_reduce_add(x_group_sum);
            if (id_local == 0) {
                xg_sum[i] = x_group_sum / SG_SIZE;
            }
#endif
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Phase 2: Sub-group cooperative GEMV with N-block tiling (n+=2) ----
    // Each subgroup processes 2 adjacent output channels.
    const int n = n_wg_start + id_sg * TILE_N;

    // Early exit for padding work-groups.
    if (n >= N_SIZE)
        return;

    const __global uchar* B0 = W + n * K_HALF_SIZE;
    const __global uchar* B1 = W + (n + 1) * K_HALF_SIZE;
    const __global half* S = Scale + n;
#if HAS_ZP
#if ZP_IS_U8
    const __global uchar* Z = ZP + n;
#else
    const __global uchar* Z = ZP + n / 2;
#endif
#endif

    float sum_all0 = 0;
    float sum_all1 = 0;

    unroll_for(int gk = 0; gk < K_SIZE / FAKE_GROUP_SIZE; gk++) {
        // Hoist scale per quantization group boundary.
        int scale_offset = (gk * FAKE_GROUP_SIZE / GROUP_SIZE) * N_SIZE;
        half s0 = S[scale_offset];
        half s1 = S[scale_offset + 1];
#if HAS_ZP
#if ZP_IS_U8
        int zp_offset = (gk * FAKE_GROUP_SIZE / GROUP_SIZE) * N_SIZE;
        half z_hf0 = convert_half(Z[zp_offset]);
        half z_hf1 = convert_half(Z[zp_offset + 1]);
#else
        int zp_offset = (gk * FAKE_GROUP_SIZE / GROUP_SIZE) * N_SIZE / 2;
        uchar z_byte = Z[zp_offset];
        half z_hf0 = convert_half(z_byte & 0xf);
        half z_hf1 = convert_half(z_byte >> 4);
#endif
#endif

#if SG_SIZE == 32
        // SUBGROUP_SIZE=32: half4 activation (4 elems × 32 lanes = 128 elems/chunk)
        half2 sum0, sum1;
        half4 a = as_half4(intel_sub_group_block_read_us4(
            (const __local ushort*)x_slm + gk * FAKE_GROUP_SIZE));
        uchar2 b  = intel_sub_group_block_read_uc2(
            (const __global uchar*)B0 + gk * FAKE_GROUP_SIZE / 2);
        uchar2 b2 = intel_sub_group_block_read_uc2(
            (const __global uchar*)B1 + gk * FAKE_GROUP_SIZE / 2);

        sum0.s0 = fma(a.s0, DEQUANT_4BIT_LO(b.s0),  (half)0);
        sum0.s1 = fma(a.s1, DEQUANT_4BIT_LO(b.s1),  (half)0);
        sum0.s0 = fma(a.s2, DEQUANT_4BIT_HI(b.s0),  sum0.s0);
        sum0.s1 = fma(a.s3, DEQUANT_4BIT_HI(b.s1),  sum0.s1);

        sum1.s0 = fma(a.s0, DEQUANT_4BIT_LO(b2.s0), (half)0);
        sum1.s1 = fma(a.s1, DEQUANT_4BIT_LO(b2.s1), (half)0);
        sum1.s0 = fma(a.s2, DEQUANT_4BIT_HI(b2.s0), sum1.s0);
        sum1.s1 = fma(a.s3, DEQUANT_4BIT_HI(b2.s1), sum1.s1);

#if HAS_ZP
        sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
        sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#else
        sum_all0 += (sum0[0] + sum0[1]) * s0;
        sum_all1 += (sum1[0] + sum1[1]) * s1;
#endif

#else  // SG_SIZE == 16
        // SUBGROUP_SIZE=16: half8 activation (8 elems × 16 lanes = 128 elems/chunk)
        half4 sum0, sum1;
        half8 a = as_half8(intel_sub_group_block_read_us8(
            (const __local ushort*)x_slm + gk * FAKE_GROUP_SIZE));
        uchar4 b  = intel_sub_group_block_read_uc4(
            (const __global uchar*)B0 + gk * FAKE_GROUP_SIZE / 2);
        uchar4 b2 = intel_sub_group_block_read_uc4(
            (const __global uchar*)B1 + gk * FAKE_GROUP_SIZE / 2);

        sum0.s0 = fma(a.s0, DEQUANT_4BIT_LO(b.s0),  (half)0);
        sum0.s1 = fma(a.s1, DEQUANT_4BIT_LO(b.s1),  (half)0);
        sum0.s2 = fma(a.s2, DEQUANT_4BIT_LO(b.s2),  (half)0);
        sum0.s3 = fma(a.s3, DEQUANT_4BIT_LO(b.s3),  (half)0);

        sum0.s0 = fma(a.s4, DEQUANT_4BIT_HI(b.s0),  sum0.s0);
        sum0.s1 = fma(a.s5, DEQUANT_4BIT_HI(b.s1),  sum0.s1);
        sum0.s2 = fma(a.s6, DEQUANT_4BIT_HI(b.s2),  sum0.s2);
        sum0.s3 = fma(a.s7, DEQUANT_4BIT_HI(b.s3),  sum0.s3);

        sum1.s0 = fma(a.s0, DEQUANT_4BIT_LO(b2.s0), (half)0);
        sum1.s1 = fma(a.s1, DEQUANT_4BIT_LO(b2.s1), (half)0);
        sum1.s2 = fma(a.s2, DEQUANT_4BIT_LO(b2.s2), (half)0);
        sum1.s3 = fma(a.s3, DEQUANT_4BIT_LO(b2.s3), (half)0);

        sum1.s0 = fma(a.s4, DEQUANT_4BIT_HI(b2.s0), sum1.s0);
        sum1.s1 = fma(a.s5, DEQUANT_4BIT_HI(b2.s1), sum1.s1);
        sum1.s2 = fma(a.s6, DEQUANT_4BIT_HI(b2.s2), sum1.s2);
        sum1.s3 = fma(a.s7, DEQUANT_4BIT_HI(b2.s3), sum1.s3);

#if HAS_ZP
        sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
        sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#else
        sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3]) * s0;
        sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3]) * s1;
#endif

#endif  // SG_SIZE
    }

    // ---- Single sub_group_reduce_add at the end ----
    sum_all0 = sub_group_reduce_add(sum_all0);
    sum_all1 = sub_group_reduce_add(sum_all1);

    // ---- Single-lane write (lane 0 only) ----
    if (id_local == 0) {
        C[b * N_SIZE + n]     = TO_OUTPUT_TYPE(sum_all0);
        if (n + 1 < N_SIZE)
            C[b * N_SIZE + n + 1] = TO_OUTPUT_TYPE(sum_all1);
    }
}

#undef unroll_for

// ============================================================
// Branch B2: i8-activation + int4-weight WOQ GEMV  (W4A8)
// N-parallel high-occupancy, same approach as B1.
// ============================================================
#else  // IS_ACT_INT8 == 1

KERNEL(gemm_generate_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE*  A,         // i8/char activation [B, K]
    const __global uchar*        W,         // u4/i4 packed weight [N, K/2]
    const __global INPUT2_TYPE*  Scale,     // f16 weight scale, fbyx: [NUM_GROUPS, N]
#if HAS_ZP
    const __global uchar*        ZP,        // ZP fbyx
#endif
    const __global half*         ActScale,  // f16 per-token activation scale [B]
    __global OUTPUT_TYPE*        C          // f16 output [B, N]
)
{
    // ---- DISPATCH: one work-item per output channel ----
    const int n = (int)get_global_id(0);
    const int b = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    // ---- HOIST base pointers ----
    const __global char*  A_row = (const __global char*)(A + b * K_SIZE);
    const __global uchar* W_row = W + n * K_HALF_SIZE;

    float acc = 0;

    // ---- CORE COMPUTE: iterate K in GROUP_SIZE chunks, hoist scale/ZP per group ----
    const int num_groups = K_SIZE / GROUP_SIZE;
    const int iters_per_group = GROUP_SIZE / VEC_SIZE;

    for (int g = 0; g < num_groups; g++) {
        const float s = (float)Scale[g * N_SIZE + n];
        float z = 0;
#if HAS_ZP
        LOAD_ZP(g, n, ZP, z);
#endif

        float group_sum = 0;
#if HAS_ZP
        float a_group_sum = 0;
#endif

        int k = g * GROUP_SIZE;
        const __global char*  a_ptr = A_row + k;
        const __global uchar* w_ptr = W_row + k / 2;

        for (int i = 0; i < iters_per_group; i++, a_ptr += VEC_SIZE, w_ptr += VEC_SIZE / 2) {
            // LOAD activation: 8 contiguous i8 values, convert to float.
            char8 ai = vload8(0, a_ptr);
            float8 a = convert_float8(ai);

            // LOAD weight: 4 bytes = 8 nibbles via single uint read.
            uint packed = *(const __global uint*)(w_ptr);
            float8 wf;
            UNPACK8_UINT(packed, wf);

            // DOT PRODUCT: fma chain.
            group_sum += a.s0 * wf.s0 + a.s1 * wf.s1
                       + a.s2 * wf.s2 + a.s3 * wf.s3
                       + a.s4 * wf.s4 + a.s5 * wf.s5
                       + a.s6 * wf.s6 + a.s7 * wf.s7;

#if HAS_ZP
            a_group_sum += a.s0 + a.s1 + a.s2 + a.s3
                         + a.s4 + a.s5 + a.s6 + a.s7;
#endif
        }

        // ACCUMULATE with dequant: scale * (dot - activation_sum * zp).
#if HAS_ZP
        acc += (group_sum - a_group_sum * z) * s;
#else
        acc += group_sum * s;
#endif
    }

    // ---- OUTPUT STORE: multiply by per-token activation scale ----
    float act_scale = (float)ActScale[b];
    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(acc * act_scale);
}

#endif  // IS_ACT_INT8

#endif  // IS_WEIGHT_INT4
