// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimised GEMV for the LLM token-generation phase (M=1).
//
// This file contains two kernel specialisations selected at JIT-compile time via
// the IS_WEIGHT_INT4 define:
//
//  IS_WEIGHT_INT4 == 0 (default)  →  pure f16/f16 GEMM
//    Inputs:  A[B,K] f16, B[B,N,K] f16
//    Dispatch: global=[ceil(N/SG)*SG, B, 1], local=[SG,1,1]
//    Each lane computes one output element with vectorised float32 dot product.
//
//  IS_WEIGHT_INT4 == 1  →  f16-activation + int4-weight GEMV with WOQ
//    Inputs:
//      INPUT0 A[B,K]         f16  activations
//      INPUT1 W[B,N,K/2]     uchar u4/i4-packed weights (low nibble = even K)
//      INPUT2 S[B,NG,N]      half  per-group per-output-channel scale
//      INPUT3 ZP[B,NG,N/2]   uchar per-group ZP (optional, u4 packed)
//        where NG = K / GROUP_SIZE = NUM_GROUPS
//    Dequant formula per group gk:
//      acc += scale[gk,n] * Σ_{k in group} A[k] * (w_int4[n,k] − ZP[gk,n])
//    ZP defaults to 8 (for symmetric u4 where range is 0–15 centred at 8).
//    For i4 weights (WEIGHT_IS_SIGNED=1) nibbles are sign-extended before subtract.

#include "include/batch_headers/common.cl"

// Horizontal sum of a float8.
#define HSUM8(v) ((v).s0 + (v).s1 + (v).s2 + (v).s3 + \
                  (v).s4 + (v).s5 + (v).s6 + (v).s7)

// ============================================================
// Branch A: f16/f16 pure GEMM
// ============================================================
#if !defined(IS_WEIGHT_INT4) || (IS_WEIGHT_INT4 == 0)

// Vector type aliases for VEC_SIZE-wide loads.
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
// For u4: value range 0-15. For i4 (WEIGHT_IS_SIGNED=1): sign-extend 0-7/8-15→8-(-1).
#if WEIGHT_IS_SIGNED
// i4 sign-extension: top nibble bit signals negative.
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
// u4: nibble as unsigned 0-15.
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

// Input tensor flat-index helpers.
// After update_impl_params the layouts are:
//   INPUT0: activation [B, K]      (INPUT0_TYPE = half)
//   INPUT1: weight     [N, K/2]    bytes  (data_type u4/i4, B folded into B dim above M)
//   INPUT2: scale      [NG, N]     (INPUT2_TYPE = half)  NG = NUM_GROUPS
//   INPUT3: ZP         [NG, N/2]   bytes  (optional, present only if HAS_ZP)
//
// Note: after update_impl_params the B batch dim is already folded into M=1, so
// for a single-query batch B==1 and the weight row-offset simplifies to n*K_HALF_SIZE.
#define K_HALF_SIZE (K_SIZE / 2)

KERNEL(gemm_generate_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE*  A,         // f16 activation [B, K]
    const __global uchar*        W,         // u4/i4 packed weight [N, K/2]
    const __global INPUT2_TYPE*  Scale,     // f16 scale [NUM_GROUPS, N]
#if HAS_ZP
    const __global uchar*        ZP,        // u4 packed ZP [NUM_GROUPS, N/2]
#endif
    __global OUTPUT_TYPE*        C          // f16 output [B, N]
)
{
    const int n = (int)get_global_id(0);   // output feature index
    const int b = (int)get_global_id(1);   // batch index

    if (n >= N_SIZE)
        return;

    // Activation base: row b of the A matrix.
    const int a_base = b * K_SIZE;
    // Weight row n: each row has K_HALF_SIZE bytes.
    const int w_base = n * K_HALF_SIZE;

    float acc = 0.0f;

    // Outer loop: iterate over quantisation groups.
    for (int gk = 0; gk < NUM_GROUPS; gk++) {
        // Load per-group dequantisation scale (one per output channel per group).
        float scale = (float)Scale[gk * N_SIZE + n];

        // Load per-group ZP (one per output channel per group).
#if HAS_ZP
        // ZP is packed u4: 2 values per byte. Channel n: even→low nibble, odd→high nibble.
        uchar zp_byte = ZP[gk * (N_SIZE / 2) + n / 2];
        float zp = (n % 2 == 0) ? (float)(zp_byte & 0xF) : (float)(zp_byte >> 4);
#else
        // Default ZP for unsigned 4-bit: 8 (centres the range 0-15 around zero).
#if WEIGHT_IS_SIGNED
        float zp = 0.0f;  // i4 symmetric: ZP=0
#else
        float zp = 8.0f;  // u4 asymmetric: ZP=8
#endif
#endif  // HAS_ZP

        // Inner loop: vectorised dot product over one group of GROUP_SIZE K-elements.
        const int k_start = gk * GROUP_SIZE;
        const int k_end   = k_start + GROUP_SIZE;  // GROUP_SIZE must equal SG_SIZE * VEC_SIZE

        float8 group_acc = (float8)(0.f);
        for (int k = k_start; k < k_end; k += VEC_SIZE) {
            // Load VEC_SIZE=8 f16 activation elements.
            half8 a_vec = vload8(0, A + a_base + k);

            // Load 4 bytes = VEC_SIZE=8 u4/i4 nibbles from the weight row.
            uchar4 w_packed = vload4(0, W + w_base + k / 2);

            // Unpack nibbles into a float8.
            float8 w_f;
            UNPACK8(w_packed, w_f);

            // Subtract ZP (same value for all 8 positions within the group on this iter).
            w_f = w_f - zp;

            // float32 fused multiply-add accumulation.
            group_acc = mad(convert_float8(a_vec), w_f, group_acc);
        }

        // Horizontal sum of the 8-wide vector accumulator, then apply scale.
        acc += HSUM8(group_acc) * scale;
    }

    // Write output (convert float → f16).
    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(acc);
}

#endif  // IS_WEIGHT_INT4

