// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimised GEMV for the LLM token-generation phase (M=1).
//
// N-parallel dispatch: each work-item computes one output element,
// iterating over the full K dimension.
// Dispatch: global=[ceil(N/SG)*SG, B, 1], local=[SG,1,1]
//
//  IS_WEIGHT_INT4 == 0  ->  pure f16/f16 GEMM
//  IS_WEIGHT_INT4 == 1  ->  W4A16 or W4A8

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

// ============================================================
// Branch B1: f16-activation + int4-weight WOQ GEMV  (W4A16)
// SLM-optimised: activations cached in shared local memory.
// Each work-group of WG_SIZE items shares one SLM activation tile
// and computes WG_SIZE * TILE_N output channels in parallel.
// Inner loop processes 16 K-elements per iteration for better ILP.
// ============================================================
#if !defined(IS_ACT_INT8) || (IS_ACT_INT8 == 0)

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
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
    const int lid = (int)get_local_id(0);
    const int wg_id = (int)get_group_id(0);
    const int b   = (int)get_global_id(1);

    // Each work-group handles WG_SIZE * TILE_N consecutive output channels.
    const int n_base = wg_id * (WG_SIZE * TILE_N) + lid * TILE_N;

    // --- Cooperatively load the full activation vector into SLM. ---
    __local INPUT0_TYPE slm_act[K_SIZE];
    const int a_base = b * K_SIZE;
    for (int i = lid; i < K_SIZE; i += WG_SIZE) {
        slm_act[i] = A[a_base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if TILE_N == 2
    // Compute 2 output channels per work-item.
    const int n0 = n_base;
    const int n1 = n_base + 1;
    const bool valid0 = (n0 < N_SIZE);
    const bool valid1 = (n1 < N_SIZE);

    if (!valid0)
        return;

    const int w_base0 = n0 * K_HALF_SIZE;
    const int w_base1 = n1 * K_HALF_SIZE;
    __global const uint* W0_uint = (__global const uint*)(W + w_base0);
    __global const uint* W1_uint = (__global const uint*)(W + w_base1);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int gk = 0; gk < NUM_GROUPS; gk++) {
        float scale0 = (float)Scale[gk * N_SIZE + n0];
        float scale1 = valid1 ? (float)Scale[gk * N_SIZE + n1] : 0.0f;

        float zp0, zp1;
        LOAD_ZP(gk, n0, ZP, zp0);
        if (valid1) { LOAD_ZP(gk, n1, ZP, zp1); } else { zp1 = 0.0f; }

        const int k_start = gk * GROUP_SIZE;
        float8 gacc0_lo = (float8)(0.f);
        float8 gacc0_hi = (float8)(0.f);
        float8 gacc1_lo = (float8)(0.f);
        float8 gacc1_hi = (float8)(0.f);

        __attribute__((opencl_unroll_hint(4)))
        for (int k = k_start; k < k_start + GROUP_SIZE; k += 16) {
            // Load 16 activations (two half8).
            half8 a_lo = vload8(0, slm_act + k);
            half8 a_hi = vload8(0, slm_act + k + 8);
            float8 af_lo = convert_float8(a_lo);
            float8 af_hi = convert_float8(a_hi);

            // Channel 0: uint reads + bitshift unpack.
            uint p0_lo = W0_uint[k / 8];
            uint p0_hi = W0_uint[k / 8 + 1];
            float8 wf0_lo, wf0_hi;
            UNPACK8_UINT(p0_lo, wf0_lo);
            UNPACK8_UINT(p0_hi, wf0_hi);
            gacc0_lo = mad(af_lo, wf0_lo - zp0, gacc0_lo);
            gacc0_hi = mad(af_hi, wf0_hi - zp0, gacc0_hi);

            // Channel 1.
            if (valid1) {
                uint p1_lo = W1_uint[k / 8];
                uint p1_hi = W1_uint[k / 8 + 1];
                float8 wf1_lo, wf1_hi;
                UNPACK8_UINT(p1_lo, wf1_lo);
                UNPACK8_UINT(p1_hi, wf1_hi);
                gacc1_lo = mad(af_lo, wf1_lo - zp1, gacc1_lo);
                gacc1_hi = mad(af_hi, wf1_hi - zp1, gacc1_hi);
            }
        }

        acc0 += (HSUM8(gacc0_lo) + HSUM8(gacc0_hi)) * scale0;
        acc1 += (HSUM8(gacc1_lo) + HSUM8(gacc1_hi)) * scale1;
    }

    C[b * N_SIZE + n0] = TO_OUTPUT_TYPE(acc0);
    if (valid1)
        C[b * N_SIZE + n1] = TO_OUTPUT_TYPE(acc1);

#else  // TILE_N == 1 (default path)

    const int n = n_base;
    if (n >= N_SIZE)
        return;

    const int w_base = n * K_HALF_SIZE;
    __global const uint* W_uint = (__global const uint*)(W + w_base);
    float acc = 0.0f;

    for (int gk = 0; gk < NUM_GROUPS; gk++) {
        float scale = (float)Scale[gk * N_SIZE + n];

        float zp;
        LOAD_ZP(gk, n, ZP, zp);

        const int k_start = gk * GROUP_SIZE;
        float8 gacc0 = (float8)(0.f);
        float8 gacc1 = (float8)(0.f);

        // Process 16 K-elements per iteration with dual accumulators for ILP.
        // Uses uint reads (4 bytes) + bitshift unpack for efficient nibble extraction.
        __attribute__((opencl_unroll_hint(4)))
        for (int k = k_start; k < k_start + GROUP_SIZE; k += 16) {
            half8 a_lo = vload8(0, slm_act + k);
            half8 a_hi = vload8(0, slm_act + k + 8);

            uint p0 = W_uint[k / 8];
            uint p1 = W_uint[k / 8 + 1];
            float8 wf0, wf1;
            UNPACK8_UINT(p0, wf0);
            UNPACK8_UINT(p1, wf1);
            wf0 -= zp;
            wf1 -= zp;
            gacc0 = mad(convert_float8(a_lo), wf0, gacc0);
            gacc1 = mad(convert_float8(a_hi), wf1, gacc1);
        }

        acc += (HSUM8(gacc0) + HSUM8(gacc1)) * scale;
    }

    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(acc);
#endif  // TILE_N
}

// ============================================================
// Branch B2: i8-activation + int4-weight WOQ GEMV  (W4A8)
// SLM-optimised: activations cached in shared local memory.
// ============================================================
#else  // IS_ACT_INT8 == 1

__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
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
    const int lid = (int)get_local_id(0);
    const int wg_id = (int)get_group_id(0);
    const int b   = (int)get_global_id(1);

    const int n_base = wg_id * (WG_SIZE * TILE_N) + lid * TILE_N;

    // --- Cooperatively load the full activation vector into SLM. ---
    __local INPUT0_TYPE slm_act[K_SIZE];
    const int a_base = b * K_SIZE;
    for (int i = lid; i < K_SIZE; i += WG_SIZE) {
        slm_act[i] = A[a_base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int n = n_base;
    if (n >= N_SIZE)
        return;

    const int w_base = n * K_HALF_SIZE;
    float acc = 0.0f;

    for (int gk = 0; gk < NUM_GROUPS; gk++) {
        float scale = (float)Scale[gk * N_SIZE + n];

        float zp;
        LOAD_ZP(gk, n, ZP, zp);

        const int k_start = gk * GROUP_SIZE;
        float8 group_acc = (float8)(0.f);

        __attribute__((opencl_unroll_hint(4)))
        for (int k = k_start; k < k_start + GROUP_SIZE; k += 16) {
            char8 a_lo = vload8(0, slm_act + k);
            char8 a_hi = vload8(0, slm_act + k + 8);

            uchar4 wp_lo = vload4(0, W + w_base + k / 2);
            uchar4 wp_hi = vload4(0, W + w_base + k / 2 + 4);
            float8 wf_lo, wf_hi;
            UNPACK8(wp_lo, wf_lo);
            UNPACK8(wp_hi, wf_hi);
            wf_lo = wf_lo - zp;
            wf_hi = wf_hi - zp;
            group_acc = mad(convert_float8(a_lo), wf_lo, group_acc);
            group_acc = mad(convert_float8(a_hi), wf_hi, group_acc);
        }

        acc += HSUM8(group_acc) * scale;
    }

    float act_scale = (float)ActScale[b];
    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(acc * act_scale);
}

#endif  // IS_ACT_INT8

#endif  // IS_WEIGHT_INT4
