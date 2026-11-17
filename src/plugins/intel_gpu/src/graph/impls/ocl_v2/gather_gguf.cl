// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Row-selective dequantizing gather over a raw GGUF block-quantised embedding table.
//
// Produces  OUT[row, :] = dequantize(WEIGHT[token_indices[row], :])
// where WEIGHT is a [VOCAB_SIZE, HIDDEN_SIZE] matrix stored as raw GGUF blocks (consumed directly
// from HBM, never materialised to dense f16). One work-item decodes exactly one GGUF block
// (GGUF_BLOCK_ELEM logical elements) of one output row, mirroring the canonical ggml reference
// (ggml-quants.c) and the CPU reference in the GGUF frontend
// (src/frontends/gguf/src/builders/dequantize.cpp) so the decode is bit-for-bit identical.
//
// The packed GGUF source format is selected at JIT time by exactly one GGUF_IS_<TYPE> flag together
// with GGUF_BLOCK_ELEM / GGUF_BLOCK_BYTES / HIDDEN_SIZE / BLOCKS_PER_ROW / ROW_BYTES.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// Reconstruct a half from two little-endian bytes (GGUF is little-endian, as is every OV target).
inline half FUNC(gguf_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
// 6-bit packed sub-block scale/min extraction shared by Q4_K / Q5_K (ggml get_scale_min_k4).
inline void FUNC(gguf_get_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
#endif

// Decode one GGUF block starting at `blk` into `vals[GGUF_BLOCK_ELEM]` in hidden order.
inline void FUNC(gguf_decode_block)(const __global uchar* blk, half* vals) {
#if defined(GGUF_IS_Q4_0)
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        const int lo = (int)(qs[j] & 0x0F) - 8;
        const int hi = (int)(qs[j] >> 4) - 8;
        vals[j] = convert_half((float)lo * d);
        vals[j + 16] = convert_half((float)hi * d);
    }
#elif defined(GGUF_IS_Q4_1)
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float m = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* qs = blk + 4;
    for (int j = 0; j < 16; ++j) {
        const float lo = (float)(qs[j] & 0x0F);
        const float hi = (float)(qs[j] >> 4);
        vals[j] = convert_half(lo * d + m);
        vals[j + 16] = convert_half(hi * d + m);
    }
#elif defined(GGUF_IS_Q8_0)
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    for (int j = 0; j < 32; ++j) {
        vals[j] = convert_half((float)qs[j] * d);
    }
#elif defined(GGUF_IS_Q4_K)
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;   // 12 bytes
    const __global uchar* qs = blk + 16;      // 128 bytes
    int o = 0;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        for (int l = 0; l < 32; ++l) {
            vals[o++] = convert_half(d1 * (float)(qs[l] & 0x0F) - m1);
        }
        for (int l = 0; l < 32; ++l) {
            vals[o++] = convert_half(d2 * (float)(qs[l] >> 4) - m2);
        }
        qs += 32;
        is += 2;
    }
#elif defined(GGUF_IS_Q5_K)
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;   // 12 bytes
    const __global uchar* qh = blk + 16;      // 32 bytes
    const __global uchar* ql = blk + 48;      // 128 bytes
    int o = 0;
    int is = 0;
    uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        for (int l = 0; l < 32; ++l) {
            const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            vals[o++] = convert_half(d1 * (float)q - m1);
        }
        for (int l = 0; l < 32; ++l) {
            const int q = (int)(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            vals[o++] = convert_half(d2 * (float)q - m2);
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
#elif defined(GGUF_IS_Q6_K)
    const __global uchar* ql = blk;                             // 128 bytes
    const __global uchar* qh = blk + 128;                       // 64 bytes
    const __global char* sc = (const __global char*)(blk + 192);// 16 signed scales
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk + 208);
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            vals[o + l + 0] = convert_half(d * (float)sc[is + 0] * (float)q1);
            vals[o + l + 32] = convert_half(d * (float)sc[is + 2] * (float)q2);
            vals[o + l + 64] = convert_half(d * (float)sc[is + 4] * (float)q3);
            vals[o + l + 96] = convert_half(d * (float)sc[is + 6] * (float)q4);
        }
        o += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
#else
#error "gather_gguf: no GGUF_IS_<TYPE> decoder selected"
#endif
}

KERNEL(gather_gguf)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global uchar* weight,
    const __global INPUT1_TYPE* token_indices,
    __global OUTPUT_TYPE* output)
{
    const uint gid = (uint)get_global_id(0);
    const uint row = gid / (uint)BLOCKS_PER_ROW;
    const uint b = gid - row * (uint)BLOCKS_PER_ROW;

    const uint token = (uint)token_indices[row];
    const __global uchar* blk = weight + (ulong)token * (ulong)ROW_BYTES + (ulong)b * (ulong)GGUF_BLOCK_BYTES;

    half vals[GGUF_BLOCK_ELEM];
    FUNC_CALL(gguf_decode_block)(blk, vals);

    const uint out_base = row * (uint)HIDDEN_SIZE + b * (uint)GGUF_BLOCK_ELEM;
    for (uint e = 0; e < (uint)GGUF_BLOCK_ELEM; ++e) {
        output[out_base + e] = vals[e];
    }
}
