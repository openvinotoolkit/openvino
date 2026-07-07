// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF weight transcode kernel (compute-bound / large-M prefill path).
//
// Converts a raw GGUF block-quantised weight matrix W[N, K] into a OneDNN-WOQ-native low-bit layout:
//   - a packed weight scratchpad: i4 (TRANSCODE_TO_I4=1) or i8 (signed), in [N, K] physical order
//     (matches dnnl wei_md [K,N] with format_tag::ba), and
//   - a parallel f16 per-group scale scratchpad [K/REQUANT_GROUP, N] = dnnl scale md [K/group, N]
//     with element (g, n) at g*N + n (per-K-group x per-N mask).
//
// The block bytes are decoded to half in registers with the SAME per-format decoders used by the
// native GEMV kernel (so numerics track exactly), then symmetrically re-quantized per REQUANT_GROUP
// elements to the target low-bit domain. dequant NEVER lands in an f16/f32 weight buffer (constraint
// C2): the only persisted weight is the low-bit scratchpad; the f16 values live only in registers.
//
// One work-item owns one (n, GGUF block): global = [N_SIZE, K_SIZE / GGUF_BLOCK_ELEM, 1], local = [SG, 1, 1].
// The block is decoded once and every REQUANT group inside it is requantized from the shared decoded
// window, so the heavy bit-unpacking runs a single time per block instead of
// (GGUF_BLOCK_ELEM / REQUANT_GROUP)x (8x for K-quants with a 256-elem block and a 32-elem group).
// REQUANT_GROUP must divide GGUF_BLOCK_ELEM (so a group never straddles two GGUF blocks).

#include "include/batch_headers/common.cl"

inline half FUNC(tq_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

// ---- per-format block decoders (identical math to fc_gguf_opt.cl) ----

#if defined(GGUF_IS_Q4_0)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        out[j]      = (half)(((int)(qs[j] & 0x0F) - 8) * (float)d);
        out[j + 16] = (half)(((int)(qs[j] >> 4)   - 8) * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q8_0)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(tq_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    for (int j = 0; j < 32; ++j) {
        out[j] = (half)((float)qs[j] * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
inline void FUNC(tq_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
#endif

#if defined(GGUF_IS_Q4_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(tq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(tq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qs     = blk + 16;
    int o = 0, is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(tq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(tq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d1 * (float)(qs[l] & 0x0F) - m1);
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d2 * (float)(qs[l] >> 4) - m2);
        qs += 32; is += 2;
    }
}
#endif

#if defined(GGUF_IS_Q5_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(tq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(tq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qh     = blk + 16;
    const __global uchar* ql     = blk + 48;
    int o = 0, is = 0; uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(tq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(tq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0); out[o++] = (half)(d1 * (float)q - m1); }
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0); out[o++] = (half)(d2 * (float)q - m2); }
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
#endif

#if defined(GGUF_IS_Q6_K)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const __global uchar* ql = blk;
    const __global uchar* qh = blk + 128;
    const __global char*  sc = (const __global char*)(blk + 192);
    const float d = (float)FUNC_CALL(tq_load_f16)(blk + 208);
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[o + l + 0]  = (half)(d * (float)sc[is + 0] * q1);
            out[o + l + 32] = (half)(d * (float)sc[is + 2] * q2);
            out[o + l + 64] = (half)(d * (float)sc[is + 4] * q3);
            out[o + l + 96] = (half)(d * (float)sc[is + 6] * q4);
        }
        o += 128; ql += 64; qh += 32; sc += 8;
    }
}
#endif

// ---- main transcode kernel ----
// TRANSCODE_TO_I4 : 1 -> pack two i4 nibbles per output byte; 0 -> one i8 per byte.
// QMAX            : 7 (i4 symmetric) or 127 (i8 symmetric).
// REQUANT_GROUP   : elements sharing one f16 scale (divides GGUF_BLOCK_ELEM).
KERNEL(fc_gguf_transcode)(
    const __global uchar* W,        // GGUF block weights [N, K] (opaque bytes)
          __global uchar* WQ,       // out: packed low-bit weight [N, K] (i4 packed / i8)
          __global half*  SC        // out: per-group f16 scale [N, K/REQUANT_GROUP]
)
{
    const int n   = (int)get_global_id(0);          // output row (subgroup lane axis, padded to SG)
    const int blk = (int)get_global_id(1);          // GGUF block index along K
    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    if (n >= N_SIZE || blk >= blocks_per_row)
        return;

    const __global uchar* w_row = W + (uint)n * (uint)blocks_per_row * GGUF_BLOCK_BYTES;

    // Decode the whole GGUF block ONCE. Every REQUANT group inside it reuses this decoded window, so
    // the expensive bit-unpacking runs a single time per block instead of once per group.
    half blk_vals[GGUF_BLOCK_ELEM];
    FUNC_CALL(tq_decode_block)(w_row + (uint)blk * GGUF_BLOCK_BYTES, blk_vals);

    const int groups_per_block = GGUF_BLOCK_ELEM / REQUANT_GROUP;
    const uint row_base = (uint)n * (uint)K_SIZE;
#if !TRANSCODE_TO_I4
    __global char* wq_i8 = (__global char*)WQ;
#endif

    // Symmetric per-group requantization for each REQUANT group within the decoded block.
    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int off_in_blk = gi * REQUANT_GROUP;        // group offset within the decoded block
        const int g  = blk * groups_per_block + gi;       // global group index along K
        const int k0 = g * REQUANT_GROUP;                 // first K element of this group

        float amax = 0.0f;
        for (int i = 0; i < REQUANT_GROUP; ++i) {
            float v = fabs((float)blk_vals[off_in_blk + i]);
            amax = fmax(amax, v);
        }
        const float scale     = (amax > 0.0f) ? (amax / (float)QMAX) : 1.0f;
        const float inv_scale = (amax > 0.0f) ? ((float)QMAX / amax) : 0.0f;

        // Scale md is [K/group, N] (per-K-group x per-N): element (g, n) at g*N + n.
        SC[(uint)g * (uint)N_SIZE + (uint)n] = (half)scale;

#if TRANSCODE_TO_I4
        // i4 packed two-per-byte; weight byte index = (n*K + k)/2. REQUANT_GROUP is even.
        for (int i = 0; i < REQUANT_GROUP; i += 2) {
            const int k = k0 + i;
            int q0 = (int)round((float)blk_vals[off_in_blk + i]     * inv_scale);
            int q1 = (int)round((float)blk_vals[off_in_blk + i + 1] * inv_scale);
            q0 = clamp(q0, -8, 7);
            q1 = clamp(q1, -8, 7);
            const uint byte_idx = (row_base + (uint)k) >> 1; // two consecutive k share one byte
            WQ[byte_idx] = (uchar)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
#else
        for (int i = 0; i < REQUANT_GROUP; ++i) {
            int q = (int)round((float)blk_vals[off_in_blk + i] * inv_scale);
            q = clamp(q, -128, 127);
            wq_i8[row_base + (uint)(k0 + i)] = (char)q;
        }
#endif
    }
}
