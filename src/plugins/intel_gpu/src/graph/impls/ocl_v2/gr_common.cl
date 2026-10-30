// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared helpers of the GatedResidual OpenCL kernels. One subgroup (16 lanes) is one Xe2 hardware thread; a row of
// 16*NV halves is held lane-interleaved: element e lives in component (e / 16) of lane (e % 16), which is exactly what
// intel_sub_group_block_read_us{1,2,4,8} returns. The header is inlined into every kernel separately (no include guard,
// helper functions go through FUNC).

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define GR_LOG2E 1.4426950408889634f

#define GR_BR_1 intel_sub_group_block_read_us
#define GR_BR_2 intel_sub_group_block_read_us2
#define GR_BR_4 intel_sub_group_block_read_us4
#define GR_BR_8 intel_sub_group_block_read_us8
#define GR_BW_1 intel_sub_group_block_write_us
#define GR_BW_2 intel_sub_group_block_write_us2
#define GR_BW_4 intel_sub_group_block_write_us4
#define GR_BW_8 intel_sub_group_block_write_us8
#define GR_CVF_1 convert_float
#define GR_CVF_2 convert_float2
#define GR_CVF_4 convert_float4
#define GR_CVF_8 convert_float8
#define GR_ASH_1 as_half
#define GR_ASH_2 as_half2
#define GR_ASH_4 as_half4
#define GR_ASH_8 as_half8
#define GR_CVH_1 convert_half_rte
#define GR_CVH_2 convert_half2_rte
#define GR_CVH_4 convert_half4_rte
#define GR_CVH_8 convert_half8_rte
#define GR_ASU_1 as_ushort
#define GR_ASU_2 as_ushort2
#define GR_ASU_4 as_ushort4
#define GR_ASU_8 as_ushort8

#define GR_UT(nv) MAKE_VECTOR_TYPE(ushort, nv)
#define GR_FT(nv) MAKE_VECTOR_TYPE(float, nv)

// ushort block of 16*nv halves -> float vector, and back (round to nearest even)
#define GR_LDF(nv, p) CAT(GR_CVF_, nv)(CAT(GR_ASH_, nv)(CAT(GR_BR_, nv)(p)))
#define GR_BRU(nv, p) CAT(GR_BR_, nv)(p)
#define GR_STF(nv, p, f) CAT(GR_BW_, nv)(p, CAT(GR_ASU_, nv)(CAT(GR_CVH_, nv)(f)))
#define GR_U2F(nv, u) CAT(GR_CVF_, nv)(CAT(GR_ASH_, nv)(u))
#define GR_F2U(nv, f) CAT(GR_ASU_, nv)(CAT(GR_CVH_, nv)(f))

// sequential per-lane accumulation over the components
#define GR_ACC_1(s, v) { s += (v); }
#define GR_ACC_2(s, v) { s += (v).s0; s += (v).s1; }
#define GR_ACC_4(s, v) { s += (v).s0; s += (v).s1; s += (v).s2; s += (v).s3; }
#define GR_ACC_8(s, v) { s += (v).s0; s += (v).s1; s += (v).s2; s += (v).s3; s += (v).s4; s += (v).s5; s += (v).s6; s += (v).s7; }
#define GR_ACC(nv, s, v) CAT(GR_ACC_, nv)(s, v)

// s += a . b over the components
#define GR_DOT_1(s, a, b) { s += (a) * (b); }
#define GR_DOT_2(s, a, b) { s += (a).s0 * (b).s0; s += (a).s1 * (b).s1; }
#define GR_DOT_4(s, a, b) { s += (a).s0 * (b).s0; s += (a).s1 * (b).s1; s += (a).s2 * (b).s2; s += (a).s3 * (b).s3; }
#define GR_DOT_8(s, a, b) { s += (a).s0 * (b).s0; s += (a).s1 * (b).s1; s += (a).s2 * (b).s2; s += (a).s3 * (b).s3; s += (a).s4 * (b).s4; s += (a).s5 * (b).s5; s += (a).s6 * (b).s6; s += (a).s7 * (b).s7; }
#define GR_DOT(nv, s, a, b) CAT(GR_DOT_, nv)(s, a, b)

// lane-interleaved SLM access of 16*nv ushorts
#define GR_SLMST_1(p, v, l) { (p)[l] = (v); }
#define GR_SLMST_2(p, v, l) { (p)[l] = (v).s0; (p)[16 + l] = (v).s1; }
#define GR_SLMST_4(p, v, l) { (p)[l] = (v).s0; (p)[16 + l] = (v).s1; (p)[32 + l] = (v).s2; (p)[48 + l] = (v).s3; }
#define GR_SLMST_8(p, v, l) { (p)[l] = (v).s0; (p)[16 + l] = (v).s1; (p)[32 + l] = (v).s2; (p)[48 + l] = (v).s3; (p)[64 + l] = (v).s4; (p)[80 + l] = (v).s5; (p)[96 + l] = (v).s6; (p)[112 + l] = (v).s7; }
#define GR_SLMST(nv, p, v, l) CAT(GR_SLMST_, nv)(p, v, l)
#define GR_SLMLD_1(p, l) ((p)[l])
#define GR_SLMLD_2(p, l) ((ushort2)((p)[l], (p)[16 + l]))
#define GR_SLMLD_4(p, l) ((ushort4)((p)[l], (p)[16 + l], (p)[32 + l], (p)[48 + l]))
#define GR_SLMLD_8(p, l) ((ushort8)((p)[l], (p)[16 + l], (p)[32 + l], (p)[48 + l], (p)[64 + l], (p)[80 + l], (p)[96 + l], (p)[112 + l]))
#define GR_SLMLD(nv, p, l) CAT(GR_SLMLD_, nv)(p, l)

#define GR_SIGMOID(x) native_recip(1.0f + native_exp2(-(x) * GR_LOG2E))

#define GR_MAD intel_sub_group_f16_f16_matrix_mad_k16

// 2D block reads of a row-major f16 matrix (surface: width/pitch in bytes, height in rows; rows past the height read 0).
// GR_RD2D(rows, ...): rows x 32 halves (two K16 slices) -> dst[s * rows + r] = M[y + r][x + 16 * s + lane]
// GR_RD2D1(rows, ...): rows x 16 halves                  -> dst[r]            = M[y + r][x + lane]
#define GR_RD2D_8 intel_sub_group_2d_block_read_16b_8r16x2c
#define GR_RD2D_16 intel_sub_group_2d_block_read_16b_16r16x2c
#define GR_RD2D_32 intel_sub_group_2d_block_read_16b_32r16x2c
#define GR_RD2D(rows, base, width, height, pitch, x, y, dst) CAT(GR_RD2D_, rows)((__global void*)(base), (width), (height), (pitch), (int2)((x), (y)), (dst))
#define GR_RD2D1_8 intel_sub_group_2d_block_read_16b_8r16x1c
#define GR_RD2D1(rows, base, width, height, pitch, x, y, dst) CAT(GR_RD2D1_, rows)((__global void*)(base), (width), (height), (pitch), (int2)((x), (y)), (dst))

// rows of 8 tokens: component i of the 8 values is token i (one SLM block message per tile)
inline void FUNC(gr_st8)(__local float* p, float8 v) {
    BLOCK_WRITEN_SLM(uint, 8, (__local uint*)p, 0, as_uint8(v));
}

inline float8 FUNC(gr_ld8)(const __local float* p) {
    return as_float8(BLOCK_READN_SLM(uint, 8, (const __local uint*)p, 0));
}
