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

// ============================================================
// Branch B1: f16-activation + int4-weight WOQ GEMV  (W4A16)
// N-parallel: each work-item computes one output channel.
// ============================================================
#if !defined(IS_ACT_INT8) || (IS_ACT_INT8 == 0)

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
    const int n = (int)get_global_id(0);
    const int b = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    const int a_base = b * K_SIZE;
    const int w_base = n * K_HALF_SIZE;

    float acc = 0.0f;

    for (int gk = 0; gk < NUM_GROUPS; gk++) {
        float scale = (float)Scale[gk * N_SIZE + n];

        float zp;
        LOAD_ZP(gk, n, ZP, zp);

        const int k_start = gk * GROUP_SIZE;
        float8 group_acc = (float8)(0.f);

        for (int k = k_start; k < k_start + GROUP_SIZE; k += VEC_SIZE) {
            half8 a_vec = vload8(0, A + a_base + k);
            uchar4 w_packed = vload4(0, W + w_base + k / 2);
            float8 w_f;
            UNPACK8(w_packed, w_f);
            w_f = w_f - zp;
            group_acc = mad(convert_float8(a_vec), w_f, group_acc);
        }

        acc += HSUM8(group_acc) * scale;
    }

    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(acc);
}

// ============================================================
// Branch B2: i8-activation + int4-weight WOQ GEMV  (W4A8)
// N-parallel: each work-item computes one output channel.
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
    const int n = (int)get_global_id(0);
    const int b = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    const int a_base = b * K_SIZE;
    const int w_base = n * K_HALF_SIZE;

    float acc = 0.0f;

    for (int gk = 0; gk < NUM_GROUPS; gk++) {
        float scale = (float)Scale[gk * N_SIZE + n];

        float zp;
        LOAD_ZP(gk, n, ZP, zp);

        const int k_start = gk * GROUP_SIZE;
        float8 group_acc = (float8)(0.f);

        for (int k = k_start; k < k_start + GROUP_SIZE; k += VEC_SIZE) {
            char8 a_vec = vload8(0, A + a_base + k);
            uchar4 w_packed = vload4(0, W + w_base + k / 2);
            float8 w_f;
            UNPACK8(w_packed, w_f);
            w_f = w_f - zp;
            group_acc = mad(convert_float8(a_vec), w_f, group_acc);
        }

        acc += HSUM8(group_acc) * scale;
    }

    float act_scale = (float)ActScale[b];
    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(acc * act_scale);
}

#endif  // IS_ACT_INT8

#endif  // IS_WEIGHT_INT4
