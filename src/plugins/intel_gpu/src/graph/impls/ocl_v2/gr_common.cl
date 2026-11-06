// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared helpers of the GatedResidual OpenCL kernels, plus the weight dequantization kernel (GR_DEQUANT_WEIGHTS). One subgroup
// (16 lanes) is one Xe2 hardware thread; a row of 16*NV halves is held lane-interleaved: element e lives in component (e / 16)
// of lane (e % 16), which is exactly what intel_sub_group_block_read_us{1,2,4,8} returns. The header is inlined into every
// kernel separately (no include guard, helper functions go through FUNC).

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

// K permutation of the 4-bit decode GEMMs (GR_KPERM), applied inside every 8 consecutive K values: storage position p holds
// k = GR_KP(p), k sits at position GR_KQ(k)
#define GR_KP(p) (((p) & ~7u) | (((p) & 7u) >> 1) | (((p) & 1u) << 2))
#define GR_KQ(k) (((k) & ~7u) | (((k) & 3u) << 1) | (((k) >> 2) & 1u))
#ifndef GR_KPERM
#define GR_KPERM 0
#endif

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

// SLM block write; the __local overload of intel_sub_group_block_write needs cl_intel_subgroup_local_block_io, otherwise each lane stores
// its own column
#if defined(cl_intel_subgroup_local_block_io)
    #define GR_BLOCK_WRITEN_SLM(type, nv, ptr, offset, val) BLOCK_WRITEN_SLM(type, nv, ptr, offset, val)
#else
    #define GR_BLOCK_WRITEN_SLM(type, nv, ptr, offset, val) GR_SLMST(nv, (__local type*)(ptr) + (offset), val, get_sub_group_local_id())
#endif

// rows of 8 tokens: component i of the 8 values is token i (one SLM block message per tile)
inline void FUNC(gr_st8)(__local float* p, float8 v) {
    GR_BLOCK_WRITEN_SLM(uint, 8, (__local uint*)p, 0, as_uint8(v));
}

inline float8 FUNC(gr_ld8)(const __local float* p) {
    return as_float8(BLOCK_READN_SLM(uint, 8, (const __local uint*)p, 0));
}

// Compressed-weight (i4/u4 and i8/u8, group size a multiple of 32, optional zero point) helpers of the GatedResidual dpas kernels.
//
// A dpas B operand is b[j] = pack(B[2j][n = lane], B[2j+1][n = lane]): the two k values of one lane share a dword. W is
// [N, K] with K contiguous, so a lane that owns row n reads its raw bytes and unpacks them in registers:
//   4-bit: one byte = (w[n,k], w[n,k+1]) for even k (low nibble = even k)  -> one VNNI dword per byte
//   8-bit: one dword = two VNNI dwords
// Signed weights are moved to the unsigned domain by flipping the sign bit of every element when the raw dwords are loaded
// (q + 2^(bits - 1)); zero points get the same shift, so q - zp is unchanged. A missing zero point is 0 (2^(bits - 1) once shifted).
// 0x6400 | v (v < 1024) is the half 1024 + v, so a pair becomes two halves with one mask/or; (pair - (1024 + zp)) * scale then
// equals half(half(q - zp) * scale) of the decompression graph bit for bit (q - zp is exact in half, one RNE multiply).
// scale / zp are [G, N] with N contiguous: one sub-group block read gives every lane the scale of its own output channel.
// A group spans GS32 K32 slices (group size 32 * GS32; the GR_W*_GS32 JIT constants); GR_W*_PERCH marks a single group.
//
// JIT of the including kernel: GR_W_TYPE (0 f16, 1 4-bit, 2 8-bit), GR_W_SIGNED (default 0), GR_HAS_ZP (default 1; kernels still
// take the zp argument, the host binds any buffer when there is no zp tensor and it is never read).

#ifndef GR_GATE_ONLY
#define GR_GATE_ONLY 0
#endif

#ifndef GR_W_TYPE
#define GR_W_TYPE 0
#endif
#ifndef GR_W_SIGNED
#define GR_W_SIGNED 0
#endif
#ifndef GR_HAS_ZP
#define GR_HAS_ZP 1
#endif
#define GR_WT_F16 0
#define GR_WT_U4 1  // 4-bit weights, signed or not (GR_W_SIGNED)
#define GR_WT_U8 2  // 8-bit weights, signed or not

#if GR_W_TYPE != GR_WT_F16

#if GR_W_TYPE == GR_WT_U4
    #define GR_WQ_BITS 4
    #define GR_WQ_SIGN 0x8u
    #define GR_WQ_RAW_FLIP 0x88888888u
#else
    #define GR_WQ_BITS 8
    #define GR_WQ_SIGN 0x80u
    #define GR_WQ_RAW_FLIP 0x80808080u
#endif
#if GR_W_SIGNED
    #define GR_WQ_FLIP(v) ((v) ^ GR_WQ_SIGN)
#else
    #define GR_WQ_FLIP(v) (v)
#endif

// Element i of a [G, N] zero point table (4-bit: packed two per byte, low nibble = even index; N % 16 == 0, so a 16-lane block
// never straddles a row) in the unsigned domain.
#if GR_HAS_ZP
    #if GR_W_TYPE == GR_WT_U4
        #define GR_WQ_ZPAT(zp, i) GR_WQ_FLIP((uint)((((zp)[(size_t)(i) >> 1]) >> (((size_t)(i) & 1) * 4)) & 0xF))
    #else
        #define GR_WQ_ZPAT(zp, i) GR_WQ_FLIP((uint)((zp)[i]))
    #endif
    // the same for an already loaded zp byte, `sh` selecting the nibble of a 4-bit table
    #if GR_W_TYPE == GR_WT_U4
        #define GR_WQ_ZPLD(b, sh) GR_WQ_FLIP((uint)(((b) >> (sh)) & 0xF))
    #else
        #define GR_WQ_ZPLD(b, sh) GR_WQ_FLIP((uint)(b))
    #endif
#else
    #define GR_WQ_ZPAT(zp, i) GR_WQ_FLIP(0u)
    #define GR_WQ_ZPLD(b, sh) GR_WQ_FLIP(0u)
#endif

// Group of K32 slice s when a group spans gs32 slices (group size = 32 * gs32)
#define GR_WQ_G(s, gs32) ((s) / (gs32))
// Offset the group-fold B operand carries: half(1024 + q) for 4-bit, half(q) for 8-bit
#if GR_W_TYPE == GR_WT_U4
    #define GR_WQ_BIAS 1024.0f
#else
    #define GR_WQ_BIAS 0.0f
#endif

#define GR_WQ_MAGIC 0x64006400u

// u4: s holds k = 8w .. 8w+7; byte b is the pair (8w+2b, 8w+2b+1), the even k in the low half
#define GR_U4_PAIR0(s) ((((s)) & 0x0000000Fu) | (((s) << 12) & 0x000F0000u) | GR_WQ_MAGIC)
#define GR_U4_PAIR1(s) ((((s) >> 8) & 0x0000000Fu) | (((s) << 4) & 0x000F0000u) | GR_WQ_MAGIC)
#define GR_U4_PAIR2(s) ((((s) >> 16) & 0x0000000Fu) | (((s) >> 4) & 0x000F0000u) | GR_WQ_MAGIC)
#define GR_U4_PAIR3(s) ((((s) >> 24) & 0x0000000Fu) | (((s) >> 12) & 0x000F0000u) | GR_WQ_MAGIC)
// u8: s holds k = 4w .. 4w+3
#define GR_U8_PAIR0(s) ((((s)) & 0x000000FFu) | (((s) << 8) & 0x00FF0000u) | GR_WQ_MAGIC)
#define GR_U8_PAIR1(s) ((((s) >> 16) & 0x000000FFu) | (((s) >> 8) & 0x00FF0000u) | GR_WQ_MAGIC)

// zp2 = (half2)(1024 + zp), sc2 = (half2)(scale): both halves of a pair belong to the same output channel
#define GR_WQ_DQ(pair, zp2, sc2) as_uint((as_half2(pair) - (zp2)) * (sc2))

// One K32 slice (q = the raw dwords of the lane's row starting at index o) -> the 16 dwords of the B operand
// (.lo = K16 slice 0, .hi = K16 slice 1)
#define GR_Q4_4(a, zp2, sc2) GR_WQ_DQ(GR_U4_PAIR0(a), zp2, sc2), GR_WQ_DQ(GR_U4_PAIR1(a), zp2, sc2), GR_WQ_DQ(GR_U4_PAIR2(a), zp2, sc2), GR_WQ_DQ(GR_U4_PAIR3(a), zp2, sc2)
#define GR_Q8_2(a, zp2, sc2) GR_WQ_DQ(GR_U8_PAIR0(a), zp2, sc2), GR_WQ_DQ(GR_U8_PAIR1(a), zp2, sc2)

#if GR_W_TYPE == GR_WT_U4
    #define GR_WQ_RAW_PER_K32 4
    #define GR_WQ_DEQ32(q, o, zp2, sc2) ((uint16)(GR_Q4_4((q)[(o)], zp2, sc2), GR_Q4_4((q)[(o) + 1], zp2, sc2), GR_Q4_4((q)[(o) + 2], zp2, sc2), GR_Q4_4((q)[(o) + 3], zp2, sc2)))
    // B operand half(1024 + q) of one K32 slice, no arithmetic: the caller applies (1024 + zp) and the group scale to the accumulator
    #define GR_U4M_4(a) GR_U4_PAIR0(a), GR_U4_PAIR1(a), GR_U4_PAIR2(a), GR_U4_PAIR3(a)
    #define GR_WQ_U4M32(q, o) ((uint16)(GR_U4M_4((q)[(o)]), GR_U4M_4((q)[(o) + 1]), GR_U4M_4((q)[(o) + 2]), GR_U4M_4((q)[(o) + 3])))
#else
    #define GR_WQ_RAW_PER_K32 8
    #define GR_WQ_DEQ32(q, o, zp2, sc2)                                                                                                 \
        ((uint16)(GR_Q8_2((q)[(o)], zp2, sc2), GR_Q8_2((q)[(o) + 1], zp2, sc2), GR_Q8_2((q)[(o) + 2], zp2, sc2), GR_Q8_2((q)[(o) + 3], zp2, sc2), \
                  GR_Q8_2((q)[(o) + 4], zp2, sc2), GR_Q8_2((q)[(o) + 5], zp2, sc2), GR_Q8_2((q)[(o) + 6], zp2, sc2), GR_Q8_2((q)[(o) + 7], zp2, sc2)))
#endif

// raw[] <- nraw dwords of the lane's weight row at K index k (brow: the row's first byte), signed weights shifted to unsigned
#if GR_W_SIGNED
#define GR_WQ_LOAD_RAW(raw, nraw, brow, k) { MAKE_VECTOR_TYPE(uint, nraw) t_ = CAT(vload, nraw)(0, (const __global uint*)((brow) + (size_t)(k) * GR_WQ_BITS / 8)); t_ ^= GR_WQ_RAW_FLIP; CAT(vstore, nraw)(t_, 0, raw); }
#else
#define GR_WQ_LOAD_RAW(raw, nraw, brow, k) { MAKE_VECTOR_TYPE(uint, nraw) t_ = CAT(vload, nraw)(0, (const __global uint*)((brow) + (size_t)(k) * GR_WQ_BITS / 8)); CAT(vstore, nraw)(t_, 0, raw); }
#endif

// scale / zp row g holds N entries; nb..nb+15 is one contiguous block read, lane l gets output channel nb + l
#define GR_WQ_SCALE2(scale, g, N, nb) ((half2)(as_half(intel_sub_group_block_read_us((scale) + (size_t)(g) * (N) + (nb)))))
#define GR_WQ_ZP2(zp, g, N, nb, lane) ((half2)((half)1024.0h + convert_half(GR_WQ_ZPAT(zp, (size_t)(g) * (N) + (nb) + (lane)))))

// 8-bit decode GEMMs (see GR_WQ_FOLD) feed half(q) as the B operand; q < 256 is exact in half
#define GR_WQ_ONES_U 0x3C003C00u
#define GR_WQ_ONES ((int8)(GR_WQ_ONES_U))
#define GR_WQ_U8H16(q, o) as_uint8(convert_half16(as_uchar16((uint4)((q)[(o)], (q)[(o) + 1], (q)[(o) + 2], (q)[(o) + 3]))))
#define GR_WQ_U8B32(q, o) ((uint16)(GR_WQ_U8H16(q, o), GR_WQ_U8H16(q, (o) + 4)))
// B operand of the group-fold loops: half(bias + q) of one K32 slice, no arithmetic (GR_WQ_BIAS)
#if GR_W_TYPE == GR_WT_U4
    #define GR_WQ_GB32(q, o) GR_WQ_U4M32(q, o)
#else
    #define GR_WQ_GB32(q, o) GR_WQ_U8B32(q, o)
#endif
// 4-bit B operand in the K-permuted order (GR_KPERM): dword j of a raw dword holds nibbles (j, j + 4), one shift and one mask
// per B dword. The activations are stored with the same permutation: position p holds k = GR_KP(p).
#define GR_U4P_4(a) (((a) & 0x000F000Fu) | GR_WQ_MAGIC), ((((a) >> 4) & 0x000F000Fu) | GR_WQ_MAGIC), ((((a) >> 8) & 0x000F000Fu) | GR_WQ_MAGIC), ((((a) >> 12) & 0x000F000Fu) | GR_WQ_MAGIC)
#define GR_WQ_U4P32(q, o) ((uint16)(GR_U4P_4((q)[(o)]), GR_U4P_4((q)[(o) + 1]), GR_U4P_4((q)[(o) + 2]), GR_U4P_4((q)[(o) + 3])))
#define GR_WQ_SCALEF(scale, nb) convert_float(as_half(intel_sub_group_block_read_us((scale) + (nb))))
#define GR_WQ_ZPF(zp, nb, lane) convert_float(GR_WQ_ZPAT(zp, (nb) + (lane)))

#endif  // GR_W_TYPE != GR_WT_F16

// Per-channel 8-bit weights (the kernel's JIT sets GR_WQ_FOLD) have their zero point and scale folded out of the K loop: the dpas
// runs on half(q), also accumulates the token row sums against a ones operand, and the epilogue applies
// acc = scale * (acc - zp * rowsum). Every other layout applies them per group (see the group-fold loops).
#ifndef GR_WQ_FOLD
#define GR_WQ_FOLD 0
#endif

#if GR_DEQUANT_WEIGHTS
// Dequantizes one compressed GatedResidual weight matrix [N, K] (4-bit / 8-bit, scale / zp [G, N]) into a
// row-major f16 scratch matrix with the arithmetic of the decompression graph, see the helpers above. The f16 GEMM kernels then
// run on the scratch (used above the GPU_GR_PD_MIN_ROWS row count).
//
// One subgroup owns 16 rows x GR_DQ_KCH columns; lane l owns row 16 * tile + l.
// JIT: GR_DEQUANT_WEIGHTS, GR_W_TYPE, GR_DQ_N, GR_DQ_K, GR_DQ_KCH, GR_DQ_NRAW (dwords per raw load), GR_DQ_RB (row bytes), GR_DQ_GS32
// (group size / 32), GR_DQ_PERCH (one group).
// Dispatch: gws = [N / 16 * 16, K / GR_DQ_KCH, 1], lws = [16, 1, 1].

#if GR_W_TYPE == GR_WT_F16
    #error "gr_common.cl: compressed weights only"
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(gr_dequant_weights)(const __global uchar* w, const __global ushort* scale, const __global uchar* zp, __global ushort* out) {
    const uint lane = get_sub_group_local_id();
    const uint n0 = get_group_id(0) * 16;
    const uint k = get_group_id(1) * GR_DQ_KCH;
    const __global uchar* brow = w + (size_t)(n0 + lane) * GR_DQ_RB;

    uint rw[GR_DQ_NRAW];
    GR_WQ_LOAD_RAW(rw, GR_DQ_NRAW, brow, k);
#if GR_DQ_PERCH
    const half2 scv = GR_WQ_SCALE2(scale, 0, GR_DQ_N, n0);
    const half2 zpv = GR_WQ_ZP2(zp, 0, GR_DQ_N, n0, lane);
#endif
    unroll_for (uint s_ = 0; s_ < GR_DQ_KCH / 32; s_++) {
#if !GR_DQ_PERCH
        const half2 scv = GR_WQ_SCALE2(scale, GR_WQ_G((k >> 5) + s_, GR_DQ_GS32), GR_DQ_N, n0);
        const half2 zpv = GR_WQ_ZP2(zp, GR_WQ_G((k >> 5) + s_, GR_DQ_GS32), GR_DQ_N, n0, lane);
#endif
        const uint16 b = GR_WQ_DEQ32(rw, s_ * GR_WQ_RAW_PER_K32, zpv, scv);
        // the B operand dwords are (k, k + 1) pairs in k order, i.e. already the row-major f16 layout
        vstore16(b, 0, (__global uint*)(out + (size_t)(n0 + lane) * GR_DQ_K + k + 32 * s_));
    }
}
#endif  // GR_DEQUANT_WEIGHTS

#if GR_DEQUANT_INJECT
// Dequantizes the compressed write-gate weight W_inject [N = C, K = C*D] (4-bit / 8-bit, scale / zp [G, N]) into a
// row-major f16 scratch matrix with the arithmetic of the decompression graph: half(half(q - zp) * scale). W_inject has only C rows,
// so there is no 16-row tile to share a scale block read; one work-item converts one element.
// JIT: GR_DEQUANT_INJECT, GR_W_TYPE, GR_DQ_N (C), GR_DQ_K (C*D), GR_DQ_GS (group size along K). Dispatch: gws = [N * K], lws = [16].
#if GR_W_TYPE == GR_WT_F16
    #error "gr_common.cl: compressed weights only"
#endif

KERNEL(gr_dequant_inject)(const __global uchar* w, const __global ushort* scale, const __global uchar* zp, __global ushort* out) {
    const uint i = get_global_id(0);
    const uint n = i / GR_DQ_K;
    const uint k = i % GR_DQ_K;
    const uint zi = (k / GR_DQ_GS) * GR_DQ_N + n;  // [G, N]
#if GR_W_TYPE == GR_WT_U4
    const uint q = GR_WQ_FLIP((w[i >> 1] >> ((i & 1) * 4)) & 0xF);
#else
    const uint q = GR_WQ_FLIP((uint)w[i]);
#endif
    const uint z = GR_WQ_ZPAT(zp, zi);
    // q - zp is exact in half; one RNE multiply by the scale, as in the decompression graph
    const half d = convert_half((int)q - (int)z);
    out[i] = as_ushort(d * as_half(scale[zi]));
}
#endif  // GR_DEQUANT_INJECT
