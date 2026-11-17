// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF prefill GEMM (OCL dpas) for Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and Q8_0 (ported from the CM gemm_q*_full.cm kernels).
//
// One source for all six formats: the host defines exactly one GG_FMT_* macro per kernel and gives each format its
// own entry point (fc_gguf_gemm_<fmt>). Each section below is the per-format tile codec (gg_fetch / gg_decode, see
// fc_gguf_gemm_body.cl) that decodes the SG-shuffled weight layout; the shared pipeline lives in the body.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "fc_gguf_gemm_common.cl"

#if GG_FMT_Q4_0
// GGUF Q4_0 prefill GEMM (port of cm/gguf/gemm_q40_full.cm). w = (q - 8) * d.
// Weights (shuffle layout): pqs_T(128B/block) || pd_T(16B/block).

#define GG_RAW_T uint8

inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 128 + (ulong)blk * 256 + j * 32), 0);
    return (uint8)(q, d, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s4));
    FUNC_CALL(gg_decode_nib4)(r.s0123, d, d * 8.0f, blo, bhi);
}

#elif GG_FMT_Q4_1
// GGUF Q4_1 prefill GEMM (port of cm/gguf/gemm_q41_full.cm). w = q * d + m.
// Weights (shuffle layout): pqs_T(128B/block) || pd_T(16B/block) || pm_T(16B/block).

#define GG_RAW_T uint8

inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const ulong meta = (ulong)blk * 256 + j * 32;
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 128 + meta), 0);
    const uint m = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 144 + meta), 0);
    return (uint8)(q, d, m, 0, 0);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s4));
    const float m = convert_float(as_half((ushort)r.s5));
    FUNC_CALL(gg_decode_nib4)(r.s0123, d, -m, blo, bhi);
}

#elif GG_FMT_Q4_K
// GGUF Q4_K prefill GEMM (port of cm/gguf/gemm_q4k_full.cm). w = q * (sq * d) - (mq * dmin).
// Weights (shuffle layout): pqs_T(128B/block) || psl_T(16B/block, SoA fields sl,ml,sh,mh,d,dmin).

#define GG_RAW_T uint8

// Raw tile: q(4) | sl | ml | sh,mh | d,dmin.
inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const __global uchar* psl = W + (ulong)num_blocks * 128 + (ulong)blk * 256;
    const uint2 slml = BLOCK_READN(uint, 2, psl, 0);
    const uint2 meta = as_uint2(BLOCK_READN(ushort, 4, (psl + 128), 0));
    return (uint8)(q, slml, meta);
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const uint sh = r.s6 & 0xFFFFu;
    const uint mh = r.s6 >> 16;
    const float d = convert_float(as_half((ushort)(r.s7 & 0xFFFFu)));
    const float dmin = convert_float(as_half((ushort)(r.s7 >> 16)));
    const uint sq = ((r.s4 >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
    const uint mq = ((r.s5 >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
    FUNC_CALL(gg_decode_nib4)(r.s0123, (float)sq * d, (float)mq * dmin, blo, bhi);
}

#elif GG_FMT_Q5_K
// GGUF Q5_K prefill GEMM (port of cm/gguf/gemm_q5k_full.cm). w = (nib | bit5 << 4) * (sq * d) - (mq * dmin).
// Weights (shuffle layout): pqs_T(128B/block) || pqh_T(32B/block) || psl_T(16B/block).

#define GG_RAW_T uint16

// Raw tile: q(4) | H | sl | ml | sh,mh | d,dmin.
inline uint16 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const uint H = BLOCK_READN(uint, 1, (W + (ulong)num_blocks * 128 + (ulong)blk * 512 + j * 64), 0);
    const __global uchar* psl = W + (ulong)num_blocks * 160 + (ulong)blk * 256;
    const uint2 slml = BLOCK_READN(uint, 2, psl, 0);
    const uint2 meta = as_uint2(BLOCK_READN(ushort, 4, (psl + 128), 0));
    return (uint16)(q, H, slml, meta, 0, 0, 0, 0, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint16 r, uint j, uint8* blo, uint8* bhi) {
    const uint H = r.s4;
    const uint sh = r.s7 & 0xFFFFu;
    const uint mh = r.s7 >> 16;
    const float d = convert_float(as_half((ushort)(r.s8 & 0xFFFFu)));
    const float dmin = convert_float(as_half((ushort)(r.s8 >> 16)));
    const uint sq = ((r.s5 >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
    const uint mq = ((r.s6 >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
    const float scale = (float)sq * d;
    const float minv = (float)mq * dmin;
    // Byte b of chunk c: low nibble K=4c+b with 5th bit at H bit 8b+c, high nibble K=16+4c+b with H bit 8b+4+c.
    uint2 lo[4];
    uint2 hi[4];
    __attribute__((opencl_unroll_hint))
    for (uint c = 0; c < 4; ++c) {
        const uint u = c == 0 ? r.s0 : c == 1 ? r.s1 : c == 2 ? r.s2 : r.s3;
        lo[c] = FUNC_CALL(gg_cvt4u)((u & 0x0F0F0F0Fu) | (((H >> c) & 0x01010101u) << 4), scale, -minv);
        hi[c] = FUNC_CALL(gg_cvt4u)(((u >> 4) & 0x0F0F0F0Fu) | (((H >> (c + 4)) & 0x01010101u) << 4), scale, -minv);
    }
    *blo = (uint8)(lo[0], lo[1], lo[2], lo[3]);
    *bhi = (uint8)(hi[0], hi[1], hi[2], hi[3]);
}

#elif GG_FMT_Q6_K
// GGUF Q6_K prefill GEMM (port of cm/gguf/gemm_q6k_full.cm). w = ((nib | h2 << 4) - 32) * (d * sc).
// Weights (shuffle layout): pql_T(128B/block) || pqh_T(64B/block) || ps_T(16B/block) || pd_T(2B/block).

#define GG_RAW_T uint8

// Raw tile: q(4) | HL | HH | d | scales(lo, hi bytes).
inline uint8 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint4 q = BLOCK_READN(uint, 4, (W + (ulong)blk * 2048 + j * 256), 0);
    const uint2 h = BLOCK_READN(uint, 2, (W + (ulong)num_blocks * 128 + (ulong)blk * 1024 + (2 * j) * 64), 0);
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 208 + (ulong)blk * 32), 0);
    const uchar2 sc = BLOCK_READN(uchar, 2, W, (ulong)num_blocks * 192 + (ulong)blk * 256 + (2 * j) * 16);
    return (uint8)(q, h, d, (uint)as_ushort(sc));
}

inline void FUNC(gg_decode)(uint8 r, uint j, uint8* blo, uint8* bhi) {
    const uint HL = r.s4;
    const uint HH = r.s5;
    const float d = convert_float(as_half((ushort)r.s6));
    const float clo = d * (float)(char)(r.s7 & 0xFFu);
    const float chi = d * (float)(char)(r.s7 >> 8);
    // Byte b of chunk c: low nibble K=4c+b with upper bits at HL bits 8b+2c, high nibble K=16+4c+b with HH bits 8b+2c.
    uint2 lo[4];
    uint2 hi[4];
    __attribute__((opencl_unroll_hint))
    for (uint c = 0; c < 4; ++c) {
        const uint u = c == 0 ? r.s0 : c == 1 ? r.s1 : c == 2 ? r.s2 : r.s3;
        lo[c] = FUNC_CALL(gg_cvt4u)((u & 0x0F0F0F0Fu) | (((HL >> (2 * c)) & 0x03030303u) << 4), clo, -32.0f * clo);
        hi[c] = FUNC_CALL(gg_cvt4u)(((u >> 4) & 0x0F0F0F0Fu) | (((HH >> (2 * c)) & 0x03030303u) << 4), chi, -32.0f * chi);
    }
    *blo = (uint8)(lo[0], lo[1], lo[2], lo[3]);
    *bhi = (uint8)(hi[0], hi[1], hi[2], hi[3]);
}

#elif GG_FMT_Q8_0
// GGUF Q8_0 prefill GEMM (port of cm/gguf/gemm_q80_full.cm). w = q * d.
// Weights (shuffle layout): pqs_T(256B/block) || pd_T(16B/block). Sub-block payload: 512B = lo(256B) || hi(256B).

#define GG_RAW_T uint16

// Raw tile: q lo(4) | q hi(4) | d.
inline uint16 FUNC(gg_fetch)(const __global uchar* W, uint num_blocks, uint blk, uint j) {
    const uint8 q = BLOCK_READN(uint, 8, (W + (ulong)blk * 4096 + j * 512), 0);
    const uint d = BLOCK_READN(ushort, 1, (W + (ulong)num_blocks * 256 + (ulong)blk * 256 + j * 32), 0);
    return (uint16)(q, d, 0, 0, 0, 0, 0, 0, 0);
}

inline void FUNC(gg_decode)(uint16 r, uint j, uint8* blo, uint8* bhi) {
    const float d = convert_float(as_half((ushort)r.s8));
    *blo = (uint8)(FUNC_CALL(gg_cvt4s)(r.s0, d), FUNC_CALL(gg_cvt4s)(r.s1, d), FUNC_CALL(gg_cvt4s)(r.s2, d), FUNC_CALL(gg_cvt4s)(r.s3, d));
    *bhi = (uint8)(FUNC_CALL(gg_cvt4s)(r.s4, d), FUNC_CALL(gg_cvt4s)(r.s5, d), FUNC_CALL(gg_cvt4s)(r.s6, d), FUNC_CALL(gg_cvt4s)(r.s7, d));
}

#else
#error "fc_gguf_gemm_opt.cl: define exactly one GG_FMT_* format macro"
#endif

#include "fc_gguf_gemm_body.cl"
