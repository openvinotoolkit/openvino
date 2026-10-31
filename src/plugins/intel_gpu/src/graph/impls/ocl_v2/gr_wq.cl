// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compressed-weight (asymmetric u4 group 32 / u8 per-channel) helpers of the GatedResidual dpas kernels. Like gr_common.cl
// this header has no include guard (every kernel inlines it separately).
//
// A dpas B operand is b[j] = pack(B[2j][n = lane], B[2j+1][n = lane]): the two k values of one lane share a dword. W is
// [N, K] with K contiguous, so a lane that owns row n reads its raw bytes and unpacks them in registers:
//   u4: one byte = (w[n,k], w[n,k+1]) for even k (low nibble = even k)  -> one VNNI dword per byte
//   u8: one dword = two VNNI dwords
// 0x6400 | v (v < 1024) is the half 1024 + v, so a pair becomes two halves with one mask/or; (pair - (1024 + zp)) * scale then
// equals half(half(q - zp) * scale) of the decompression graph bit for bit (q - zp is exact in half, one RNE multiply).
// scale / zp are [G, N] with N contiguous: one sub-group block read gives every lane the scale of its own output channel.
//
// JIT of the including kernel: GR_W_TYPE (0 f16, 1 u4, 2 u8).

#ifndef GR_W_TYPE
#define GR_W_TYPE 0
#endif
#define GR_WT_F16 0
#define GR_WT_U4 1
#define GR_WT_U8 2

#if GR_W_TYPE != GR_WT_F16

#if GR_W_TYPE == GR_WT_U4
    #define GR_WQ_BITS 4
#else
    #define GR_WQ_BITS 8
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
#else
    #define GR_WQ_RAW_PER_K32 8
    #define GR_WQ_DEQ32(q, o, zp2, sc2)                                                                                                 \
        ((uint16)(GR_Q8_2((q)[(o)], zp2, sc2), GR_Q8_2((q)[(o) + 1], zp2, sc2), GR_Q8_2((q)[(o) + 2], zp2, sc2), GR_Q8_2((q)[(o) + 3], zp2, sc2), \
                  GR_Q8_2((q)[(o) + 4], zp2, sc2), GR_Q8_2((q)[(o) + 5], zp2, sc2), GR_Q8_2((q)[(o) + 6], zp2, sc2), GR_Q8_2((q)[(o) + 7], zp2, sc2)))
#endif

// raw[] <- nraw dwords of the lane's weight row at K index k (brow: the row's first byte)
#define GR_WQ_LOAD_RAW(raw, nraw, brow, k) { MAKE_VECTOR_TYPE(uint, nraw) t_ = CAT(vload, nraw)(0, (const __global uint*)((brow) + (size_t)(k) * GR_WQ_BITS / 8)); CAT(vstore, nraw)(t_, 0, raw); }

// scale / zp row g holds N entries; nb..nb+15 is one contiguous block read, lane l gets output channel nb + l
#define GR_WQ_SCALE2(scale, g, N, nb) ((half2)(as_half(intel_sub_group_block_read_us((scale) + (size_t)(g) * (N) + (nb)))))
#define GR_WQ_ZP2(zp, g, N, nb, lane) ((half2)((half)1024.0h + convert_half((zp)[(size_t)(g) * (N) + (nb) + (lane)])))

// u8 decode GEMMs (see GR_WQ_FOLD) feed half(q) as the B operand; q < 256 is exact in half
#define GR_WQ_ONES ((int8)(0x3C003C00))
#define GR_WQ_U8H16(q, o) as_uint8(convert_half16(as_uchar16((uint4)((q)[(o)], (q)[(o) + 1], (q)[(o) + 2], (q)[(o) + 3]))))
#define GR_WQ_U8B32(q, o) ((uint16)(GR_WQ_U8H16(q, o), GR_WQ_U8H16(q, (o) + 4)))
#define GR_WQ_SCALEF(scale, nb) convert_float(as_half(intel_sub_group_block_read_us((scale) + (nb))))
#define GR_WQ_ZPF(zp, nb, lane) convert_float((zp)[(nb) + (lane)])

#endif  // GR_W_TYPE != GR_WT_F16

// u8 is per-channel, so its zero point and scale are folded out of the K loop: the dpas runs on half(q), also accumulates
// the token row sums against a ones operand, and the epilogue applies acc = scale * (acc - zp * rowsum).
#define GR_WQ_FOLD (GR_W_TYPE == GR_WT_U8)
