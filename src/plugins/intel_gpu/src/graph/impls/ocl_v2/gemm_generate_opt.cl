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
//      INPUT0 A[B,K]              f16  activations
//      INPUT1 W[N,K/2]            uchar u4/i4-packed weights (low nibble = even K)
//      INPUT2 S, fbyx:[f=N, b=NG]  half  per-group per-output-channel scale
//      INPUT3 ZP, fbyx (optional, present only if HAS_ZP):
//               u8: [f=N, b=NG]    1 byte/entry
//               u4: [f=N, b=NG/2]  2 nibbles/byte along group dim
//        where NG = K / GROUP_SIZE = NUM_GROUPS
//    Scale/ZP use fbyx memory format: f(=N=6144) is outer, b(=NG=32) is inner.
//      scale(n, gk) = Scale[n * NUM_GROUPS + gk]
//    Dequant formula per group gk:
//      acc += scale[gk,n] * Σ_{k in group} A[k] * (w_int4[n,k] − ZP[gk,n])
//    ZP defaults to 0 when no explicit ZP tensor is provided (HAS_ZP=0).
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
//   INPUT0: activation [B, K]      (f16 for W4A16, char/i8 for W4A8)
//   INPUT1: weight     [N, K/2]    bytes  (data_type u4/i4, B folded into B dim above M)
//   INPUT2: scale      [NG, N]     (INPUT2_TYPE = half)  NG = NUM_GROUPS
//   INPUT3: ZP         [N, NG/2]   bytes  (optional, present only if HAS_ZP)
//   (W4A8 only) next input: ActScale [B]  half per-token activation scale
//
// Note: after update_impl_params the B batch dim is already folded into M=1, so
// for a single-query batch B==1 and the weight row-offset simplifies to n*K_HALF_SIZE.
#define K_HALF_SIZE (K_SIZE / 2)

// -----------------------------------------------------------------------
// Shared ZP dequant helper macro used by both the f16 and i8 activation paths.
// Emits a 'float zp' variable for the given group index gk and output channel n.
// -----------------------------------------------------------------------
#if HAS_ZP
// Scale and ZP tensors use fbyx memory format.
// Tensor shape printed as "fbyx:6144x32" means:
//   f = 6144 = N (output channels) — OUTERMOST dim (slowest in memory)
//   b = 32  = NUM_GROUPS           — INNER dim (fastest in memory)
// Physical stride for f: NUM_GROUPS (=32);  stride for b: 1.
// So element at (output_channel=n, group=gk):
//   scale(n, gk)  = Scale[n * NUM_GROUPS + gk]   f16, 1 element/entry
//   zp_u8(n, gk)  = ZP   [n * NUM_GROUPS + gk]   u8,  1 byte/entry
//   zp_u4(n, gk): nibbles packed along the b (group) dimension:
//                  byte = ZP[n * (NUM_GROUPS/2) + gk/2],  nibble = gk%2
#if ZP_IS_U8
// u8 ZP: one byte per (output-channel n, group gk), fbyx → n * NUM_GROUPS + gk.
#define LOAD_ZP(gk, n, ZP, zp_var)  (zp_var) = (float)ZP[(n) * NUM_GROUPS + (gk)]
#else
// u4 ZP: 2 nibbles per byte, packed along the group dimension (b in fbyx),
// so two consecutive gk values share one byte for the same output channel n.
//   byte  = ZP[n * (NUM_GROUPS/2) + gk/2]
//   nibble= gk % 2  (low nibble = even gk, high nibble = odd gk)
#define LOAD_ZP(gk, n, ZP, zp_var)                                         \
    do {                                                                    \
        uchar zp_byte_ = ZP[(n) * (NUM_GROUPS / 2) + (gk) / 2];           \
        (zp_var) = ((gk) % 2 == 0) ? (float)(zp_byte_ & 0xF)              \
                                    : (float)(zp_byte_ >> 4);              \
    } while (0)
#endif  // ZP_IS_U8
#else
// When no explicit ZP tensor is present, use ZP=0 for both u4 and i4.
// This matches OneDNN's default behaviour: when no DNNL_ARG_ATTR_ZERO_POINTS
// attribute is set, OneDNN applies no zero-point adjustment (ZP=0).
// Note: if the model uses asymmetric u4 with an implicit ZP offset, it MUST
// supply an explicit ZP tensor so that HAS_ZP=1 and the branch above is taken.
#define LOAD_ZP(gk, n, ZP, zp_var)  (zp_var) = 0.0f
#endif  // HAS_ZP

// ============================================================
// Branch B1: f16-activation + int4-weight WOQ GEMV  (W4A16)
// ============================================================
#if !defined(IS_ACT_INT8) || (IS_ACT_INT8 == 0)

KERNEL(gemm_generate_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE*  A,         // f16 activation [B, K]
    const __global uchar*        W,         // u4/i4 packed weight [N, K/2]
    const __global INPUT2_TYPE*  Scale,     // f16 scale, fbyx: [NUM_GROUPS, N]
#if HAS_ZP
    const __global uchar*        ZP,        // ZP fbyx: u8=[NUM_GROUPS,N] or u4=[NUM_GROUPS,N/2]
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
        // Scale fbyx[f=N, b=NG]: physical = n * NUM_GROUPS + gk
        float scale = (float)Scale[n * NUM_GROUPS + gk];

        float zp;
        LOAD_ZP(gk, n, ZP, zp);

#if defined(DEBUG_FCCmpOpt) && (DEBUG_FCCmpOpt >= 1)
        if (b == 0 && n == 0)
            printf("[FCCmpOpt-W4A16] gk=%d scale=%.6f zp=%.2f k_start=%d\n",
                   gk, scale, zp, gk * GROUP_SIZE);
#endif

        // Inner loop: vectorised dot product over one group of GROUP_SIZE K-elements.
        const int k_start = gk * GROUP_SIZE;

        float8 group_acc = (float8)(0.f);
        for (int k = k_start; k < k_start + GROUP_SIZE; k += VEC_SIZE) {
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
#if defined(DEBUG_FCCmpOpt) && (DEBUG_FCCmpOpt >= 1)
    if (b == 0 && n < 4)
        printf("[FCCmpOpt-W4A16] b=0 n=%d acc=%.6f out=%.6f\n", n, acc, (float)TO_OUTPUT_TYPE(acc));
#endif
    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(acc);
}

// ============================================================
// Branch B2: i8-activation + int4-weight WOQ GEMV  (W4A8)
//
// Kernel inputs (positional, matching FCCompressedOptGenerator::get_arguments_desc):
//   INPUT0: i8/char activation  [B, K]
//   INPUT1: u4/i4 packed weight [N, K/2]
//   INPUT2: f16 weight scale    fbyx [f=N, b=NG]       physical: n*NG + gk
//   INPUT3: ZP                  fbyx (only if HAS_ZP):
//                                u8:  [f=N, b=NG]       1 byte/entry,   n*NG + gk
//                                u4:  [f=N, b=NG/2]     2 nibbles/byte, n*(NG/2) + gk/2, nibble=gk%2
//   next  : f16 act scale       [B]                      (per-token, always present in W4A8)
//   OUTPUT: f16 output          [B, N]
//
// Math: output[b,n] = act_scale[b] *
//         Σ_{gk} scale_w[n,gk] * Σ_{k in gk} act[b,k] * (w4[n,k] − zp_w[n,gk])
// ============================================================
#else  // IS_ACT_INT8 == 1

KERNEL(gemm_generate_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE*  A,         // i8/char activation [B, K]
    const __global uchar*        W,         // u4/i4 packed weight [N, K/2]
    const __global INPUT2_TYPE*  Scale,     // f16 weight scale, fbyx: [N, NUM_GROUPS]
#if HAS_ZP
    const __global uchar*        ZP,        // ZP fbyx: u8=[N,NUM_GROUPS] or u4=[N,NUM_GROUPS/2]
#endif
    const __global half*         ActScale,  // f16 per-token activation scale [B]
    __global OUTPUT_TYPE*        C          // f16 output [B, N]
)
{
    const int n = (int)get_global_id(0);   // output feature index
    const int b = (int)get_global_id(1);   // batch index

    if (n >= N_SIZE)
        return;

    const int a_base = b * K_SIZE;
    const int w_base = n * K_HALF_SIZE;

    // ---------------------------------------------------------------
    // Level-2 debug: one-time header dump for (b=0, n=0)
    // Prints raw tensor values so we can verify memory layout offline.
    // ---------------------------------------------------------------
#if defined(DEBUG_FCCmpOpt) && (DEBUG_FCCmpOpt >= 2)
    if (b == 0 && n == 0) {
        // Dump Scale[0..NUM_GROUPS-1] - scale row for output channel 0
        printf("[FCCmpOpt-DBG2] Scale[n=0, gk=0..3]: %.6f %.6f %.6f %.6f\n",
               (float)Scale[0], (float)Scale[1], (float)Scale[2], (float)Scale[3]);
#if HAS_ZP
        printf("[FCCmpOpt-DBG2] ZP[n=0, gk=0..3]: %.0f %.0f %.0f %.0f\n",
               (float)ZP[0], (float)ZP[1], (float)ZP[2], (float)ZP[3]);
#endif
        // Dump first 8 weight nibbles for row n=0 (bytes at W[0..3] = nibbles 0..7)
        uchar4 w0 = vload4(0, W);
        printf("[FCCmpOpt-DBG2] W raw bytes[0..3]: %d %d %d %d\n",
               (int)w0.s0, (int)w0.s1, (int)w0.s2, (int)w0.s3);
        printf("[FCCmpOpt-DBG2] W nibbles[0..7]: %d %d %d %d %d %d %d %d\n",
               (int)(w0.s0 & 0xF), (int)(w0.s0 >> 4),
               (int)(w0.s1 & 0xF), (int)(w0.s1 >> 4),
               (int)(w0.s2 & 0xF), (int)(w0.s2 >> 4),
               (int)(w0.s3 & 0xF), (int)(w0.s3 >> 4));
        // Dump first 8 activation values for b=0
        printf("[FCCmpOpt-DBG2] A[b=0, k=0..7]: %d %d %d %d %d %d %d %d\n",
               (int)A[0], (int)A[1], (int)A[2], (int)A[3],
               (int)A[4], (int)A[5], (int)A[6], (int)A[7]);
        // Dump first 4 ActScale values (check which index = current token)
        printf("[FCCmpOpt-DBG2] ActScale[0..3]: %.6f %.6f %.6f %.6f\n",
               (float)ActScale[0], (float)ActScale[1],
               (float)ActScale[2], (float)ActScale[3]);
        printf("[FCCmpOpt-DBG2] B_SIZE=%d K_SIZE=%d N_SIZE=%d NUM_GROUPS=%d GROUP_SIZE=%d\n",
               B_SIZE, K_SIZE, N_SIZE, NUM_GROUPS, GROUP_SIZE);
    }
#endif

    float acc = 0.0f;

    for (int gk = 0; gk < NUM_GROUPS; gk++) {
        float scale = (float)Scale[n * NUM_GROUPS + gk];

        float zp;
        LOAD_ZP(gk, n, ZP, zp);

#if defined(DEBUG_FCCmpOpt) && (DEBUG_FCCmpOpt >= 1)
        if (b == 0 && n == 0)
            printf("[FCCmpOpt-W4A8] gk=%d scale=%.6f zp=%.2f\n", gk, scale, zp);
#endif

        const int k_start = gk * GROUP_SIZE;
        float gk_acc = 0.0f;

        float8 group_acc = (float8)(0.f);
        for (int k = k_start; k < k_start + GROUP_SIZE; k += VEC_SIZE) {
            char8 a_vec = vload8(0, A + a_base + k);
            uchar4 w_packed = vload4(0, W + w_base + k / 2);
            float8 w_f;
            UNPACK8(w_packed, w_f);
            w_f = w_f - zp;
            group_acc = mad(convert_float8(a_vec), w_f, group_acc);
        }
        float group_dot = HSUM8(group_acc);

#if defined(DEBUG_FCCmpOpt) && (DEBUG_FCCmpOpt >= 2)
        if (b == 0 && n == 0)
            printf("[FCCmpOpt-W4A8] gk=%d group_dot=%.4f scale=%.6f contrib=%.6f\n",
                   gk, group_dot, scale, group_dot * scale);
#endif

        acc += group_dot * scale;
    }

    float act_scale = (float)ActScale[b];
    float result = acc * act_scale;

#if defined(DEBUG_FCCmpOpt) && (DEBUG_FCCmpOpt >= 1)
    if (b == 0 && n < 8)
        printf("[FCCmpOpt-W4A8] b=0 n=%d raw_acc=%.6f act_scale=%.8f result=%.6f\n",
               n, acc, act_scale, result);
#endif

    C[b * N_SIZE + n] = TO_OUTPUT_TYPE(result);
}

#endif  // IS_ACT_INT8

#endif  // IS_WEIGHT_INT4

