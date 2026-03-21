// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimised GEMV for the LLM token-generation phase (M=1).
//
// Sub-group cooperative K-split: each work-group of SG_SIZE lanes computes
// ONE output element.  All lanes split the K reduction dimension, then
// merge partial sums via sub_group_reduce_add.  This gives coalesced weight
// reads (consecutive lanes read consecutive K bytes from the same row).
//
//  IS_WEIGHT_INT4 == 0  →  pure f16/f16 GEMM
//    Dispatch: global=[N*SG_SIZE, B, 1], local=[SG_SIZE,1,1]
//
//  IS_WEIGHT_INT4 == 1  →  W4A16 or W4A8
//    Dispatch: global=[N*SG_SIZE, B, 1], local=[SG_SIZE,1,1]

#include "include/batch_headers/common.cl"

// K-elements processed per sub-group per step.
#define K_PER_STEP (SG_SIZE * VEC_SIZE)

// Default N_TILE: outputs per work-group. Must match JIT value.
#ifndef N_TILE
#define N_TILE 1
#endif

// Horizontal sum of a float8.
#define HSUM8(v) ((v).s0 + (v).s1 + (v).s2 + (v).s3 + \
                  (v).s4 + (v).s5 + (v).s6 + (v).s7)

// ============================================================
// Branch A: f16/f16 pure GEMM — sub-group cooperative K-split
// ============================================================
#if !defined(IS_WEIGHT_INT4) || (IS_WEIGHT_INT4 == 0)

#define INPUT_VEC_TYPE   MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE)
#define WEIGHT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT1_TYPE, VEC_SIZE)
#define INPUT_VLOAD      CAT(vload, VEC_SIZE)

REQD_SUB_GROUP_SIZE(SG_SIZE)
KERNEL(gemm_generate_opt)(
    const __global INPUT0_TYPE*  A,   // activations [B, K]
    const __global INPUT1_TYPE*  B,   // weights     [B, N, K]
          __global OUTPUT_TYPE*  C    // output      [B, N]
)
{
    const int n_base = (int)get_group_id(0) * N_TILE;
    const int lane   = (int)get_sub_group_local_id();
    const int b_idx  = (int)get_global_id(1);

    const int a_base = b_idx * K_SIZE;

    float lane_acc[N_TILE];
    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < N_TILE; t++)
        lane_acc[t] = 0.0f;

    for (int k_block = 0; k_block < K_SIZE; k_block += K_PER_STEP) {
        const int k = k_block + lane * VEC_SIZE;
        INPUT_VEC_TYPE a_vec = INPUT_VLOAD(0, A + a_base + k);
        float8 af = convert_float8(a_vec);

        __attribute__((opencl_unroll_hint))
        for (int t = 0; t < N_TILE; t++) {
            const int n = n_base + t;
            if (n < N_SIZE) {
                const int wb = b_idx * N_SIZE * K_SIZE + n * K_SIZE;
                WEIGHT_VEC_TYPE w_vec = INPUT_VLOAD(0, B + wb + k);
                lane_acc[t] += HSUM8(af * convert_float8(w_vec));
            }
        }
    }

    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < N_TILE; t++) {
        float result = sub_group_reduce_add(lane_acc[t]);
        const int n = n_base + t;
        if (lane == 0 && n < N_SIZE)
            C[b_idx * N_SIZE + n] = TO_OUTPUT_TYPE(result);
    }
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
// PartialShape [N, NG] maps to BFYX as [B=N, F=NG, Y=1, X=1].
// In fbyx physical layout, F(=NG) is outermost (slowest), B(=N) is inner (fastest).
//   physical stride for F(=NG): N_SIZE;  stride for B(=N): 1.
// So element at (output_channel=n, group=gk):
//   scale(n, gk)  = Scale[gk * N_SIZE + n]   f16, 1 element/entry
//   zp_u8(n, gk)  = ZP   [gk * N_SIZE + n]   u8,  1 byte/entry
//   zp_u4(n, gk): nibbles packed along the F(=gk) dimension:
//                  byte = ZP[(gk/2) * N_SIZE + n],  nibble = gk%2
#if ZP_IS_U8
// u8 ZP: one byte per (group gk, output-channel n), fbyx → gk * N_SIZE + n.
#define LOAD_ZP(gk, n, ZP, zp_var)  (zp_var) = (float)ZP[(gk) * N_SIZE + (n)]
#else
// u4 ZP: 2 nibbles per byte, packed along the F(=gk) dimension,
// so two consecutive gk values share one byte for the same output channel n.
//   byte  = ZP[(gk/2) * N_SIZE + n]
//   nibble= gk % 2  (low nibble = even gk, high nibble = odd gk)
#define LOAD_ZP(gk, n, ZP, zp_var)                                         \
    do {                                                                    \
        uchar zp_byte_ = ZP[((gk) / 2) * N_SIZE + (n)];                   \
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
//            Sub-group cooperative K-split — coalesced weight reads.
// ============================================================
#if !defined(IS_ACT_INT8) || (IS_ACT_INT8 == 0)

REQD_SUB_GROUP_SIZE(SG_SIZE)
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
    const int n_base = (int)get_group_id(0) * N_TILE;
    const int lane   = (int)get_sub_group_local_id();
    const int b      = (int)get_global_id(1);

    const int a_base = b * K_SIZE;

    float lane_acc[N_TILE];
    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < N_TILE; t++)
        lane_acc[t] = 0.0f;

    for (int k_block = 0; k_block < K_SIZE; k_block += K_PER_STEP) {
        const int k0 = k_block + lane * VEC_SIZE;
        const int gk = k0 / GROUP_SIZE;

        // Activation loaded once, reused across N_TILE tiles.
        half8  a_vec = vload8(0, A + a_base + k0);
        float8 af = convert_float8(a_vec);

        __attribute__((opencl_unroll_hint))
        for (int t = 0; t < N_TILE; t++) {
            const int n = n_base + t;
            if (n < N_SIZE) {
                const int wb = n * K_HALF_SIZE;
                float sc = (float)Scale[gk * N_SIZE + n];
                float zp;
                LOAD_ZP(gk, n, ZP, zp);

                uchar4 wp = vload4(0, W + wb + k0 / 2);
                float8 wf;
                UNPACK8(wp, wf);
                wf = wf - zp;
                lane_acc[t] += HSUM8(af * wf) * sc;
            }
        }
    }

    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < N_TILE; t++) {
        float result = sub_group_reduce_add(lane_acc[t]);
        const int n = n_base + t;
        if (lane == 0 && n < N_SIZE)
            C[b * N_SIZE + n] = TO_OUTPUT_TYPE(result);
    }
}

// ============================================================
// Branch B2: i8-activation + int4-weight WOQ GEMV  (W4A8)
//            Sub-group cooperative K-split — coalesced weight reads.
// ============================================================
#else  // IS_ACT_INT8 == 1

REQD_SUB_GROUP_SIZE(SG_SIZE)
KERNEL(gemm_generate_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE*  A,         // i8/char activation [B, K]
    const __global uchar*        W,         // u4/i4 packed weight [N, K/2]
    const __global INPUT2_TYPE*  Scale,     // f16 weight scale, fbyx: [N, NUM_GROUPS]
#if HAS_ZP
    const __global uchar*        ZP,        // ZP fbyx
#endif
    const __global half*         ActScale,  // f16 per-token activation scale [B]
    __global OUTPUT_TYPE*        C          // f16 output [B, N]
)
{
    const int n_base = (int)get_group_id(0) * N_TILE;
    const int lane   = (int)get_sub_group_local_id();
    const int b      = (int)get_global_id(1);

    const int a_base = b * K_SIZE;

    float lane_acc[N_TILE];
    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < N_TILE; t++)
        lane_acc[t] = 0.0f;

    for (int k_block = 0; k_block < K_SIZE; k_block += K_PER_STEP) {
        const int k0 = k_block + lane * VEC_SIZE;
        const int gk = k0 / GROUP_SIZE;

        char8  a_vec = vload8(0, A + a_base + k0);
        float8 af = convert_float8(a_vec);

        __attribute__((opencl_unroll_hint))
        for (int t = 0; t < N_TILE; t++) {
            const int n = n_base + t;
            if (n < N_SIZE) {
                const int wb = n * K_HALF_SIZE;
                float sc = (float)Scale[gk * N_SIZE + n];
                float zp;
                LOAD_ZP(gk, n, ZP, zp);

                uchar4 wp = vload4(0, W + wb + k0 / 2);
                float8 wf;
                UNPACK8(wp, wf);
                wf = wf - zp;
                lane_acc[t] += HSUM8(af * wf) * sc;
            }
        }
    }

    float act_scale = (float)ActScale[b];

    __attribute__((opencl_unroll_hint))
    for (int t = 0; t < N_TILE; t++) {
        float result = sub_group_reduce_add(lane_acc[t]) * act_scale;
        const int n = n_base + t;
        if (lane == 0 && n < N_SIZE)
            C[b * N_SIZE + n] = TO_OUTPUT_TYPE(result);
    }
}

#endif  // IS_ACT_INT8

#endif  // IS_WEIGHT_INT4

