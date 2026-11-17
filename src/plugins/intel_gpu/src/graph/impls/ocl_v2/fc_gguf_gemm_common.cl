// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared helpers for the GGUF prefill GEMM kernels (fc_gguf_gemm_opt.cl), ported from the CM
// kernels cm/gguf/gemm_q*_full.cm. Pure helpers only; the pipeline itself lives in
// fc_gguf_gemm_body.cl.
//
// Operand layout used with intel_sub_group_f16_f16_matrix_mad_k16 (SIMD16, M=8, N=16, K=16):
//   A  (short8): lane = K column, element i = token row i      (2D block read gives this directly)
//   B  (int8)  : lane = output column n, element sd = half2 {K=2*sd, K=2*sd+1} of that column
//   acc(float8): lane = output column n, element i = token row i

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#if defined(cl_intel_subgroup_local_block_io)
#pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#endif

// Pack two floats into a half2 (round-to-nearest-even, same as the CM float->half assignment).
inline uint FUNC(gg_pk)(float a, float b) {
    return as_uint(convert_half2_rte((float2)(a, b)));
}

// Load the activation tile [8 tokens x 32 K] starting at column x, row y as {lo: K 0..15, hi: K 16..31}.
// Rows >= M are returned as zero by the 2D block hardware.
#if GG_A_F32
inline void FUNC(gg_load_a)(const __global INPUT0_TYPE* A, uint K, uint M, int x, int y, short8* lo, short8* hi) {
    uint8 u0;
    uint8 u1;
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)A, K * 4, M, K * 4, (int2)(x, y), (private uint*)&u0);
    intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)A, K * 4, M, K * 4, (int2)(x + 16, y), (private uint*)&u1);
    *lo = as_short8(convert_half8_rte(as_float8(u0)));
    *hi = as_short8(convert_half8_rte(as_float8(u1)));
}
#else
inline void FUNC(gg_load_a)(const __global INPUT0_TYPE* A, uint K, uint M, int x, int y, short8* lo, short8* hi) {
    ushort16 u;
    intel_sub_group_2d_block_read_16b_8r16x2c((__global void*)A, K * 2, M, K * 2, (int2)(x, y), (private ushort*)&u);
    *lo = as_short8(u.lo);
    *hi = as_short8(u.hi);
}
#endif

// Load nt (2 or 4) consecutive 8-token tiles x 32 K with one 2D block read (f16 activations only).
#define GG_LOAD_A_MULTI(A, K, M, x, y, nt, alo, ahi)                                                                    \
    do {                                                                                                               \
        ushort gg_u[16 * (nt)];                                                                                        \
        if ((nt) == 4)                                                                                                 \
            intel_sub_group_2d_block_read_16b_32r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (int2)((x), (y)), gg_u); \
        else                                                                                                           \
            intel_sub_group_2d_block_read_16b_16r16x2c((__global void*)(A), (K) * 2, (M), (K) * 2, (int2)((x), (y)), gg_u); \
        __attribute__((opencl_unroll_hint)) for (uint gg_t = 0; gg_t < (nt); ++gg_t) {                                 \
            alo[gg_t] = as_short8(vload8(0, gg_u + 8 * gg_t));                                                         \
            ahi[gg_t] = as_short8(vload8(0, gg_u + 8 * (nt) + 8 * gg_t));                                              \
        }                                                                                                              \
    } while (0)

// SLM tile access: a tile is 8 x 16 uints laid out [sd][lane] so that every lane touches a distinct bank.
#if defined(cl_intel_subgroup_local_block_io) && !defined(GG_SLM_NO_BLOCK)
#define GG_SLM_BLOCK_IO 1
inline int8 FUNC(gg_slm_rd8)(const __local uint* p, uint lane) {
    return BLOCK_READN_SLM(int, 8, p, 0);
}

inline void FUNC(gg_slm_wr8)(__local uint* p, uint lane, uint8 v) {
    BLOCK_WRITEN_SLM(uint, 8, p, 0, v);
}
#else
inline int8 FUNC(gg_slm_rd8)(const __local uint* p, uint lane) {
    return as_int8((uint8)(p[lane], p[16 + lane], p[32 + lane], p[48 + lane], p[64 + lane], p[80 + lane], p[96 + lane], p[112 + lane]));
}

inline void FUNC(gg_slm_wr8)(__local uint* p, uint lane, uint8 v) {
    p[lane] = v.s0;
    p[16 + lane] = v.s1;
    p[32 + lane] = v.s2;
    p[48 + lane] = v.s3;
    p[64 + lane] = v.s4;
    p[80 + lane] = v.s5;
    p[96 + lane] = v.s6;
    p[112 + lane] = v.s7;
}
#endif

inline float FUNC(gg_f8_el)(float8 v, uint i) {
    switch (i) {
    case 0: return v.s0;
    case 1: return v.s1;
    case 2: return v.s2;
    case 3: return v.s3;
    case 4: return v.s4;
    case 5: return v.s5;
    case 6: return v.s6;
    default: return v.s7;
    }
}

// Four unsigned bytes (consecutive K) -> two half2 words of byte * s + o.
inline uint2 FUNC(gg_cvt4u)(uint bytes, float s, float o) {
    const float4 f = mad(convert_float4(as_uchar4(bytes)), (float4)(s), (float4)(o));
    return (uint2)(FUNC_CALL(gg_pk)(f.s0, f.s1), FUNC_CALL(gg_pk)(f.s2, f.s3));
}

// Four signed bytes (consecutive K) -> two half2 words of byte * s.
inline uint2 FUNC(gg_cvt4s)(uint bytes, float s) {
    const float4 f = convert_float4(as_char4(bytes)) * s;
    return (uint2)(FUNC_CALL(gg_pk)(f.s0, f.s1), FUNC_CALL(gg_pk)(f.s2, f.s3));
}

// Q4-style decode of one sub-block (32 K) for this lane: element = nibble * scale - minv.
// q holds 4 chunks; chunk c byte b low nibble is K = 4c+b, high nibble is K = 16+4c+b.
inline void FUNC(gg_decode_nib4)(uint4 q, float scale, float minv, uint8* blo, uint8* bhi) {
    const uint2 l0 = FUNC_CALL(gg_cvt4u)(q.s0 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 l1 = FUNC_CALL(gg_cvt4u)(q.s1 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 l2 = FUNC_CALL(gg_cvt4u)(q.s2 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 l3 = FUNC_CALL(gg_cvt4u)(q.s3 & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h0 = FUNC_CALL(gg_cvt4u)((q.s0 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h1 = FUNC_CALL(gg_cvt4u)((q.s1 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h2 = FUNC_CALL(gg_cvt4u)((q.s2 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    const uint2 h3 = FUNC_CALL(gg_cvt4u)((q.s3 >> 4) & 0x0F0F0F0Fu, scale, -minv);
    *blo = (uint8)(l0, l1, l2, l3);
    *bhi = (uint8)(h0, h1, h2, h3);
}
