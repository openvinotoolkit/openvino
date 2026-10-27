// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF weight transcode kernel (compute-bound / large-M prefill path).
//
// Converts a raw GGUF block-quantised weight matrix W[N, K] into a OneDNN-WOQ-native low-bit layout:
//   - a packed weight scratchpad: u4 (TRANSCODE_TO_I4=1) or u8 (unsigned asymmetric), in [N, K]
//     physical order (matches dnnl wei_md [K,N] with format_tag::ba), and
//   - a parallel f16 per-group scale scratchpad [K/REQUANT_GROUP, N] = dnnl scale md [K/group, N]
//     with element (g, n) at g*N + n (per-K-group x per-N mask), and
//   - a parallel u8 per-group zero-point scratchpad [K/REQUANT_GROUP, N] with the same indexing.
//
// The block bytes are decoded to half in registers with the SAME per-format decoders used by the
// native GEMV kernel (so numerics track exactly), then ASYMMETRICALLY re-quantized per REQUANT_GROUP
// elements to the target unsigned low-bit domain. Asymmetric quantization (u4/u8 + zero-point)
// matches the NNCF FP16-4BIT format consumed by oneDNN's jit:gemm:any W4A8 path on Xe2/B580:
//   val ≈ (q - zp) * scale,  q ∈ [0, QMAX],  zp = round(-vmin * QMAX / (vmax - vmin))
// dequant NEVER lands in an f16/f32 weight buffer (constraint C2): the only persisted weight is the
// low-bit scratchpad; the f16 values live only in registers.
//
// One work-item owns one (n, GGUF block): global = [N_SIZE, K_SIZE / GGUF_BLOCK_ELEM, 1], local = [SG, 1, 1].
// The block is decoded once and every REQUANT group inside it is requantized from the shared decoded
// window, so the heavy bit-unpacking runs a single time per block instead of
// (GGUF_BLOCK_ELEM / REQUANT_GROUP)x (8x for K-quants with a 256-elem block and a 32-elem group).
// REQUANT_GROUP must divide GGUF_BLOCK_ELEM (so a group never straddles two GGUF blocks).

#include "include/batch_headers/common.cl"
#include "gguf/gguf_iq_tables.hpp"

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

#if defined(GGUF_IS_Q4_1)
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(tq_load_f16)(blk);
    const half m = FUNC_CALL(tq_load_f16)(blk + 2);
    const __global uchar* qs = blk + 4;
    for (int j = 0; j < 16; ++j) {
        out[j]      = (half)((float)(qs[j] & 0x0F) * (float)d + (float)m);
        out[j + 16] = (half)((float)(qs[j] >> 4)   * (float)d + (float)m);
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

#if defined(GGUF_IS_Q3_K)
// Q3_K block layout (110 bytes / 256 elements; ggml block_q3_k). Same unpack as
// frontend dequantize_q3_k() and fc_gguf_opt.cl GGUF_IS_Q3_K: 2-bit payload plus
// high-mask correction -> q in {-4..3}, multiplied by per-16 signed sub-scale and d_all.
//
// Optimisations vs. the literal CPU port (numerically bit-exact: only the SCHEDULE changes,
// every individual FP op uses the same operands in the same order as the naive version):
//   * 32B hmask + 64B qs are pulled into private memory ONCE. The naive loop re-reads each
//     qs byte 4x (one per j shift) and each hmask byte 8x (4 shifts x 2 halves) from
//     __global, i.e. ~384 redundant uchar loads per block.
//   * The 12-byte packed scales are decoded into sixteen pre-scaled sub-scales (dl[0..15])
//     up front, so the inner lane loop is a flat dl[] lookup instead of a 4-way ternary on
//     `is` plus a byte-shift on a uint.
//   * Both the n (half) and j (shift) outer loops are hinted to unroll, so shift/mask/dl-index
//     are compile-time constants -> the 16-lane lane loop SIMD-vectorises cleanly.
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    // 1. Cache global block bytes once.
    __private uchar hmask[32];
    __private uchar qs[64];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 32; ++i) hmask[i] = blk[i];           // hmask = blk[0..31]
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 64; ++i) qs[i]    = blk[32 + i];      // qs    = blk[32..95]

    const float d_all = (float)FUNC_CALL(tq_load_f16)(blk + 108);

    // 2. Decode the 12-byte packed scales -> sixteen pre-scaled sub-scales.
    const uint kMask1 = 0x03030303u;
    const uint kMask2 = 0x0f0f0f0fu;
    const __global uchar* ps = blk + 96;
    uint aux0 = (uint)ps[0] | ((uint)ps[1] << 8) | ((uint)ps[2] << 16) | ((uint)ps[3] << 24);
    uint aux1 = (uint)ps[4] | ((uint)ps[5] << 8) | ((uint)ps[6] << 16) | ((uint)ps[7] << 24);
    uint aux2 = (uint)ps[8] | ((uint)ps[9] << 8) | ((uint)ps[10] << 16) | ((uint)ps[11] << 24);
    const uint tmp = aux2;
    aux2 = ((aux0 >> 4) & kMask2) | (((tmp >> 4) & kMask1) << 4);
    uint aux3 = ((aux1 >> 4) & kMask2) | (((tmp >> 6) & kMask1) << 4);
    aux0 = (aux0 & kMask2) | (((tmp >> 0) & kMask1) << 4);
    aux1 = (aux1 & kMask2) | (((tmp >> 2) & kMask1) << 4);

    __private float dl[16];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 4; ++i) {
        dl[i +  0] = d_all * (float)((int)(char)((aux0 >> (8 * i)) & 0xFFu) - 32);
        dl[i +  4] = d_all * (float)((int)(char)((aux1 >> (8 * i)) & 0xFFu) - 32);
        dl[i +  8] = d_all * (float)((int)(char)((aux2 >> (8 * i)) & 0xFFu) - 32);
        dl[i + 12] = d_all * (float)((int)(char)((aux3 >> (8 * i)) & 0xFFu) - 32);
    }

    // 3. 2 halves (n=0 / n=128) x 4 j-shifts, each writing 32 halfs.
    int o = 0;
    __attribute__((opencl_unroll_hint))
    for (int nh = 0; nh < 2; ++nh) {
        const __private uchar* qs_h = qs + 32 * nh;
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 4; ++j) {
            const int   shift = 2 * j;
            const uchar mask  = (uchar)(1u << (4 * nh + j));
            const float dl_lo = dl[8 * nh + 2 * j + 0];
            const float dl_hi = dl[8 * nh + 2 * j + 1];
            __attribute__((opencl_unroll_hint))
            for (int l = 0; l < 16; ++l) {
                const int q = (int)((qs_h[l]      >> shift) & 3) - ((hmask[l]      & mask) ? 0 : 4);
                out[o + l]      = (half)(dl_lo * (float)q);
            }
            __attribute__((opencl_unroll_hint))
            for (int l = 0; l < 16; ++l) {
                const int q = (int)((qs_h[l + 16] >> shift) & 3) - ((hmask[l + 16] & mask) ? 0 : 4);
                out[o + l + 16] = (half)(dl_hi * (float)q);
            }
            o += 32;
        }
    }
}
#endif

#if defined(GGUF_IS_IQ3_XXS)
// Inner l-loop is fully unrolled so `7*l` / `2*l` / sign-bit masks become compile-time constants.
// The 8 stores per l target independent `out[]` slots, so no accumulator-chain break is needed
// here (unlike the streaming dot in fc_gguf_opt.cl which uses 4-way acc).
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    const __global uchar* scales_signs = blk + 2 + 64;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const __global uchar* p4 = scales_signs + 4 * ib32;
        const uint  aux32 = (uint)p4[0] | ((uint)p4[1] << 8) | ((uint)p4[2] << 16) | ((uint)p4[3] << 24);
        const float db    = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const __global uchar* qsp = qs + 8 * ib32;
        const int ao_b = 32 * ib32;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 4; ++l) {
            const uchar signs = CONST_ARRAY_REF(ksigns_iq2xs)[(aux32 >> (7 * l)) & 127u];
            const uint  g1    = CONST_ARRAY_REF(iq3xxs_grid)[qsp[2 * l + 0]];
            const uint  g2    = CONST_ARRAY_REF(iq3xxs_grid)[qsp[2 * l + 1]];
            const int   ao    = ao_b + 8 * l;
            out[ao + 0] = (half)(db * (float)(uchar)( g1        & 0xFFu) * ((signs & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db * (float)(uchar)((g1 >>  8) & 0xFFu) * ((signs & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db * (float)(uchar)((g1 >> 16) & 0xFFu) * ((signs & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db * (float)(uchar)((g1 >> 24) & 0xFFu) * ((signs & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db * (float)(uchar)( g2        & 0xFFu) * ((signs & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db * (float)(uchar)((g2 >>  8) & 0xFFu) * ((signs & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db * (float)(uchar)((g2 >> 16) & 0xFFu) * ((signs & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db * (float)(uchar)((g2 >> 24) & 0xFFu) * ((signs & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

#if defined(GGUF_IS_IQ3_S)
// IQ3_S block layout (110 bytes / 256 elements; ggml block_iq3_s). See fc_gguf_opt.cl IQ3_S section
// for the field-level description. The decoded block is 256 halfs; downstream per-32 requant
// (REQUANT_GROUP=32) aligns one-to-one with the ib32 sub-block boundary (each ib32 sub-block of
// 32 elements has its own `db = d * (1 + 2*scale4)` sub-scale).

// Inner l-loop fully unrolled so `2*l` shifts collapse to compile-time constants for the
// 9-bit index assembly. Independent out[] writes -> no accumulator-chain break needed.
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* qh     = blk + 66;
    const __global uchar* signs  = blk + 74;
    const __global uchar* scales = blk + 106;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc_byte = scales[ib32 >> 1];
        const int   sc4     = (ib32 & 1) ? (int)(sc_byte >> 4) : (int)(sc_byte & 0xF);
        const float db      = d * (float)(1 + 2 * sc4);
        const uchar qhb     = qh[ib32];
        const __global uchar* qsp = qs + 8 * ib32;
        const __global uchar* sgp = signs + 4 * ib32;
        const int ao_b = 32 * ib32;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 4; ++l) {
            const uint  idx1 = (uint)qsp[2*l + 0] | ((((uint)qhb >> (2*l    )) & 1u) << 8);
            const uint  idx2 = (uint)qsp[2*l + 1] | ((((uint)qhb >> (2*l + 1)) & 1u) << 8);
            const uint  g1   = CONST_ARRAY_REF(iq3s_grid)[idx1];
            const uint  g2   = CONST_ARRAY_REF(iq3s_grid)[idx2];
            const uchar sb   = sgp[l];
            const int   ao   = ao_b + 8 * l;
            out[ao + 0] = (half)(db * (float)(uchar)( g1        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db * (float)(uchar)((g1 >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db * (float)(uchar)((g1 >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db * (float)(uchar)((g1 >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db * (float)(uchar)( g2        & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db * (float)(uchar)((g2 >>  8) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db * (float)(uchar)((g2 >> 16) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db * (float)(uchar)((g2 >> 24) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

#if defined(GGUF_IS_IQ2_S)
// IQ2_S block layout (82 bytes / 256 elements; ggml block_iq2_s). See fc_gguf_opt.cl IQ2_S section
// for the field-level description. Note: each REQUANT_GROUP=32 = one ib32 sub-block carries TWO
// independent sub-scales db0/db1 (low/high nibble of scales[ib32]); the transcode_target() maps
// IQ2_S to i8 so the worst-case ~167:1 in-group dynamic range survives requantisation.
// l-loop peeled into l=0,1 (db0 for elems 0..15) and l=2,3 (db1 for elems 16..31) so the
// per-element `(l<2) ? db0 : db1` ternary becomes a compile-time constant per peeled body.
// Bit-exact vs the literal CPU port: identical FP ops in identical order, only the loop
// structure changes (each out[] slot receives the same value, written in the same sequence).
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* signs  = blk + 34;
    const __global uchar* qh     = blk + 66;
    const __global uchar* scales = blk + 74;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc  = scales[ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + (float)(sc >> 4))  * 0.25f;
        const uchar qhb = qh[ib32];
        const __global uchar* qsp = qs    + 4 * ib32;
        const __global uchar* sgp = signs + 4 * ib32;
        const int ao_b = 32 * ib32;
        // l = 0, 1 -> elems 0..15 use db0.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 2; ++l) {
            const uint  idx = (uint)qsp[l] | (((uint)qhb << (8 - 2 * l)) & 0x300u);
            const ulong g   = CONST_ARRAY_REF(iq2s_grid)[idx];
            const uchar sb  = sgp[l];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db0 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db0 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db0 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db0 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db0 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db0 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db0 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db0 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
        // l = 2, 3 -> elems 16..31 use db1.
        __attribute__((opencl_unroll_hint))
        for (int l = 2; l < 4; ++l) {
            const uint  idx = (uint)qsp[l] | (((uint)qhb << (8 - 2 * l)) & 0x300u);
            const ulong g   = CONST_ARRAY_REF(iq2s_grid)[idx];
            const uchar sb  = sgp[l];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db1 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db1 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db1 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db1 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db1 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db1 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db1 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db1 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

#if defined(GGUF_IS_IQ2_XS)
// IQ2_XS block layout (74 bytes / 256 elements; ggml block_iq2_xs). See fc_gguf_opt.cl IQ2_XS
// section for the field-level description. Note: each REQUANT_GROUP=32 = one ib32 sub-block
// carries TWO independent sub-scales db0/db1 (low/high nibble of scales[ib32]); the
// transcode_target() maps IQ2_XS to i8 (same tier as IQ2_S) so the worst-case ~167:1 in-group
// dynamic range survives requantisation.
// l-loop peeled (same idiom as IQ2_S transcode above): l=0,1 use db0, l=2,3 use db1.
// Bit-exact: identical FP ops in identical order, only loop structure changes.
inline void FUNC(tq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d = (float)FUNC_CALL(tq_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* scales = blk + 66;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc  = scales[ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + (float)(sc >> 4))  * 0.25f;
        const __global uchar* qsp = qs + 8 * ib32;
        const int ao_b = 32 * ib32;
        // l = 0, 1 -> elems 0..15 use db0.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 2; ++l) {
            const uint  q   = (uint)qsp[2 * l] | ((uint)qsp[2 * l + 1] << 8);
            const ulong g   = CONST_ARRAY_REF(iq2xs_grid)[q & 0x1FFu];
            const uchar sb  = CONST_ARRAY_REF(ksigns_iq2xs)[(q >> 9) & 0x7Fu];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db0 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db0 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db0 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db0 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db0 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db0 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db0 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db0 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
        // l = 2, 3 -> elems 16..31 use db1.
        __attribute__((opencl_unroll_hint))
        for (int l = 2; l < 4; ++l) {
            const uint  q   = (uint)qsp[2 * l] | ((uint)qsp[2 * l + 1] << 8);
            const ulong g   = CONST_ARRAY_REF(iq2xs_grid)[q & 0x1FFu];
            const uchar sb  = CONST_ARRAY_REF(ksigns_iq2xs)[(q >> 9) & 0x7Fu];
            const int   ao  = ao_b + 8 * l;
            out[ao + 0] = (half)(db1 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            out[ao + 1] = (half)(db1 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            out[ao + 2] = (half)(db1 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            out[ao + 3] = (half)(db1 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            out[ao + 4] = (half)(db1 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            out[ao + 5] = (half)(db1 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            out[ao + 6] = (half)(db1 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            out[ao + 7] = (half)(db1 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
}
#endif

// ---- shuffle-layout prefill: decode the SG-shuffled Q4_K / Q6_K weight (fc_gguf_q4k_sg.cl /
//      fc_gguf_q6k_sg.cl layout) directly into the 256 f16 block values, then requant as usual. ----
// When GGUF_SHUFFLE, the weight Constant was reordered at compile_model into the plane-separated,
// SG-transposed layout (RepackGGUFWeightsShuffle). The transcode work-item owns one (n, blk); it reads
// its own block's shuffled bytes (per-lane 4-byte chunks) and reconstructs the 256 decoded values,
// bit-exactly matching the native tq_decode_block (only the byte source order differs). This keeps the
// single-weight-copy invariant for the prefill path.
//
// SG-shuffled layout (SG = 16, nbpr = blocks-per-row):
//   row n -> group h = n/16, lane lid = n%16.
//   Q4_K: pqs base 0, entry (h,bid) at (h*nbpr+bid)*16*128, lane lid byte o at chunk (o/4)*16*4 + lid*4 + o%4.
//         psl base N*nbpr*128, entry (h,bid) at (h*nbpr+bid)*16*16; SoA: sl@lid*4, ml@64+lid*4,
//         sh@128+lid*2, mh@160+lid*2, d@192+lid*2, dmin@224+lid*2.
//   Q6_K: pql base 0 (like pqs); pqh base N*nbpr*128, entry (h,bid) at (h*nbpr+bid)*16*64, lane lid byte
//         o at chunk (o/4)*16*4 + lid*4 + o%4; ps base +N*nbpr*64, entry (h,bid) at (h*nbpr+bid)*16*16,
//         scale si at si*16 + lid; pd base +N*nbpr*16, entry (h,bid) at (h*nbpr+bid)*16*2, half at lid*2.
#ifndef GGUF_SHUFFLE
#define GGUF_SHUFFLE 0
#endif
#ifndef TRANSCODE_ASYMMETRIC
#define TRANSCODE_ASYMMETRIC 1  // default: asymmetric (set by host based on format)
#endif
#ifndef TRANSCODE_TO_F16
#define TRANSCODE_TO_F16 0
#endif
#define SG_T 16                  // SG group width (matches RepackGGUFWeightsShuffle kSG / OPG)

#if GGUF_SHUFFLE
#pragma OPENCL EXTENSION cl_intel_subgroups       : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char  : enable

// Coalesced sub-group block gather of this SG-entry's plane bytes into private uints.
//
// This kernel is memory-bound: for the shuffle prefill the dominant cost is streaming the plane-
// separated weight bytes in from global memory. The shuffle layout stores, for 4-byte chunk c and
// lane lid, the uint at byte offset (entry + c*SG_T*4 + lid*4); i.e. across the SG_T lanes chunk c
// spans a contiguous SG_T*4-byte (SG_T-uint) region. That is EXACTLY the access pattern
// intel_sub_group_block_read consumes -> component c of a wide read8 delivers chunk (c0+c) to lane
// lid in a single hardware block-load transaction (512 B / call), replacing the previous SG_T*4
// scattered scalar uchar loads per chunk that the compiler could not coalesce. This mirrors the
// coalesced-weight-load path the sibling GEMV kernels (fc_gguf_q4k_sg.cl / fc_gguf_q6k_sg.cl) use.
//
// out_u is a private uint[n_uint]; the byte-wise decoders reinterpret it as uchar* (little-endian,
// bit-exact with the previous per-byte gather). Requires the launch sub-group width == SG_T, which
// the transcode dispatch guarantees (local = {GGUF_GEMV_SG_SIZE=16, 1, 1}; lane lid == n % SG_T).
inline void FUNC(sh_gather_plane)(const __global uchar* base, uint entry, __private uint* out_u, int n_uint) {
    const __global uint* p = (const __global uint*)(base + entry);
    int c = 0;
    // Wide 8-chunk (512 B) block reads to keep many loads in flight.
    __attribute__((opencl_unroll_hint))
    for (; c + 8 <= n_uint; c += 8) {
        const uint8 v = intel_sub_group_block_read8(p + (uint)c * (uint)SG_T);
        out_u[c + 0] = v.s0; out_u[c + 1] = v.s1; out_u[c + 2] = v.s2; out_u[c + 3] = v.s3;
        out_u[c + 4] = v.s4; out_u[c + 5] = v.s5; out_u[c + 6] = v.s6; out_u[c + 7] = v.s7;
    }
    // Tail (n_uint not a multiple of 8): one chunk per block read.
    __attribute__((opencl_unroll_hint))
    for (; c < n_uint; ++c) {
        out_u[c] = intel_sub_group_block_read(p + (uint)c * (uint)SG_T);
    }
}
#endif

#if GGUF_SHUFFLE && defined(GGUF_IS_Q4_K)
// Decode one SG-shuffled Q4_K block -> 256 halfs, matching the native GGUF_IS_Q4_K decode order.
inline void FUNC(tq_decode_shuffle_q4k)(const __private uchar* pqs, const __private uchar* psl, __private half* out) {
    const uint sl = (uint)psl[0] | ((uint)psl[1] << 8) | ((uint)psl[2] << 16) | ((uint)psl[3] << 24);
    const uint ml = (uint)psl[4] | ((uint)psl[5] << 8) | ((uint)psl[6] << 16) | ((uint)psl[7] << 24);
    const uint sh = (uint)psl[8] | ((uint)psl[9] << 8);
    const uint mh = (uint)psl[10] | ((uint)psl[11] << 8);
    const ushort d_bits = (ushort)psl[12] | ((ushort)psl[13] << 8);
    const ushort dn_bits = (ushort)psl[14] | ((ushort)psl[15] << 8);
    const float d = (float)as_half(d_bits);
    const float dmin = (float)as_half(dn_bits);
    // The shuffle stored, for sub-block j, byte pqs[j*16+k] = low(pos k) | (high(pos 16+k) << 4), and
    // the reference dequant maps sub-block j's 32 elems to K positions j*32 + {0..31}.
    for (int j = 0; j < 8; ++j) {
        const uint sq = ((sl >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
        const uint mq = ((ml >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
        const float sc = (float)sq * d;
        const float mv = (float)mq * dmin;
        const int base = j * 32;
        __attribute__((opencl_unroll_hint))
        for (int k = 0; k < 16; ++k) {
            const uchar b = pqs[j * 16 + k];
            out[base + k]      = (half)((float)(b & 0x0F) * sc - mv);
            out[base + 16 + k] = (half)((float)((b >> 4) & 0x0F) * sc - mv);
        }
    }
}
#endif

#if GGUF_SHUFFLE && defined(GGUF_IS_Q5_K)
// Decode one SG-shuffled Q5_K block -> 256 halfs, matching the native GGUF_IS_Q5_K decode order.
// Q5_K == Q4_K plus one high bit per weight. pqs/psl are read exactly like Q4_K; pqh holds, for
// sub-block j, a 4-byte word whose bit (wi/4) of byte (wi%4) is the high bit of weight wi (0..31).
inline void FUNC(tq_decode_shuffle_q5k)(const __private uchar* pqs, const __private uchar* pqh, const __private uchar* psl, __private half* out) {
    const uint sl = (uint)psl[0] | ((uint)psl[1] << 8) | ((uint)psl[2] << 16) | ((uint)psl[3] << 24);
    const uint ml = (uint)psl[4] | ((uint)psl[5] << 8) | ((uint)psl[6] << 16) | ((uint)psl[7] << 24);
    const uint sh = (uint)psl[8] | ((uint)psl[9] << 8);
    const uint mh = (uint)psl[10] | ((uint)psl[11] << 8);
    const ushort d_bits = (ushort)psl[12] | ((ushort)psl[13] << 8);
    const ushort dn_bits = (ushort)psl[14] | ((ushort)psl[15] << 8);
    const float d = (float)as_half(d_bits);
    const float dmin = (float)as_half(dn_bits);
    for (int j = 0; j < 8; ++j) {
        const uint sq = ((sl >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
        const uint mq = ((ml >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
        const float sc = (float)sq * d;
        const float mv = (float)mq * dmin;
        const uint qhw = (uint)pqh[j * 4 + 0] | ((uint)pqh[j * 4 + 1] << 8) | ((uint)pqh[j * 4 + 2] << 16) | ((uint)pqh[j * 4 + 3] << 24);
        const int base = j * 32;
        __attribute__((opencl_unroll_hint))
        for (int k = 0; k < 16; ++k) {
            const uchar b = pqs[j * 16 + k];
            // low-half weight wi=k -> byte k%4 bit k/4; high-half weight wi=16+k -> byte k%4 bit k/4+4.
            const uint lo_hb = (qhw >> ((uint)(k & 3) * 8u + (uint)(k >> 2)))        & 1u;
            const uint hi_hb = (qhw >> ((uint)(k & 3) * 8u + (uint)(k >> 2) + 4u))   & 1u;
            const int qlo = (int)((b & 0x0F)        | (lo_hb << 4));
            const int qhi = (int)(((b >> 4) & 0x0F) | (hi_hb << 4));
            out[base + k]      = (half)((float)qlo * sc - mv);
            out[base + 16 + k] = (half)((float)qhi * sc - mv);
        }
    }
}
#endif

#if GGUF_SHUFFLE && defined(GGUF_IS_Q6_K)
// Decode one SG-shuffled Q6_K block -> 256 halfs, matching the native GGUF_IS_Q6_K decode order.
// Shift table for the 16 positions in a pqh group: {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30}.
inline void FUNC(tq_decode_shuffle_q6k)(const __private uchar* pql, const __private uchar* pqh, const __private char* ps, half dh, __private half* out) {
    const int SH[16] = {0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30};
    const float d = (float)dh;
    for (int j = 0; j < 8; ++j) {
        const float clo = d * (float)ps[2 * j];
        const float chi = d * (float)ps[2 * j + 1];
        const uint hlo = (uint)pqh[j * 8 + 0] | ((uint)pqh[j * 8 + 1] << 8) | ((uint)pqh[j * 8 + 2] << 16) | ((uint)pqh[j * 8 + 3] << 24);
        const uint hhi = (uint)pqh[j * 8 + 4] | ((uint)pqh[j * 8 + 5] << 8) | ((uint)pqh[j * 8 + 6] << 16) | ((uint)pqh[j * 8 + 7] << 24);
        const int base = j * 32;
        __attribute__((opencl_unroll_hint))
        for (int k = 0; k < 16; ++k) {
            const uchar b = pql[j * 16 + k];
            const int sh = SH[k];
            const int qlo = (int)((b & 0x0F) | (((hlo >> sh) & 3u) << 4)) - 32;
            const int qhi = (int)(((b >> 4) & 0x0F) | (((hhi >> sh) & 3u) << 4)) - 32;
            out[base + k]      = (half)(clo * (float)qlo);
            out[base + 16 + k] = (half)(chi * (float)qhi);
        }
    }
}
#endif

// ---- SG-shuffle small-block (Q4_0 / Q4_1 / Q8_0) super-block decoders. EIGHT native 32-weight
//      blocks are grouped into one 256-weight super-block; the shuffle scatter (RepackGGUFWeightsShuffle)
//      is byte-identical to the sibling GEMV kernels (fc_gguf_q4_0_sg.cl / q4_1 / q8_0). pd/pm are SoA
//      fp16 per-lane fields; pqs is chunk-interleaved (128 B for 4-bit, 256 B for 8-bit). ----
#if GGUF_SHUFFLE && defined(GGUF_IS_Q4_0)
// Q4_0: w = (nibble - 8) * d_j.  pqs sub-block j byte k = qs[k] of native block j (low@k, high@16+k).
inline void FUNC(tq_decode_shuffle_q4_0)(const __private uchar* pqs, const __private half* dv, __private half* out) {
    for (int j = 0; j < 8; ++j) {
        const float d = (float)dv[j];
        const int base = j * 32;
        __attribute__((opencl_unroll_hint))
        for (int k = 0; k < 16; ++k) {
            const uchar b = pqs[j * 16 + k];
            out[base + k]      = (half)(((float)(int)(b & 0x0F) - 8.0f) * d);
            out[base + 16 + k] = (half)(((float)(int)(b >> 4)   - 8.0f) * d);
        }
    }
}
#endif

#if GGUF_SHUFFLE && defined(GGUF_IS_Q4_1)
// Q4_1: w = nibble * d_j + m_j.
inline void FUNC(tq_decode_shuffle_q4_1)(const __private uchar* pqs, const __private half* dv, const __private half* mv, __private half* out) {
    for (int j = 0; j < 8; ++j) {
        const float d = (float)dv[j];
        const float m = (float)mv[j];
        const int base = j * 32;
        __attribute__((opencl_unroll_hint))
        for (int k = 0; k < 16; ++k) {
            const uchar b = pqs[j * 16 + k];
            out[base + k]      = (half)((float)(int)(b & 0x0F) * d + m);
            out[base + 16 + k] = (half)((float)(int)(b >> 4)   * d + m);
        }
    }
}
#endif

#if GGUF_SHUFFLE && defined(GGUF_IS_Q8_0)
// Q8_0: w = q * d_j (q signed int8). pqs sub-block j = 32 int8 weights of native block j.
inline void FUNC(tq_decode_shuffle_q8_0)(const __private char* pqs, const __private half* dv, __private half* out) {
    for (int j = 0; j < 8; ++j) {
        const float d = (float)dv[j];
        const int base = j * 32;
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < 32; ++i) {
            out[base + i] = (half)((float)pqs[j * 32 + i] * d);
        }
    }
}
#endif

// ---- main transcode kernel ----
// TRANSCODE_TO_I4 : 1 -> pack two u4 nibbles per output byte; 0 -> one u8 per byte.
// QMAX            : 15 (u4 asymmetric) or 255 (u8 asymmetric).
// REQUANT_GROUP   : elements sharing one f16 scale and one u8 zero-point (divides GGUF_BLOCK_ELEM).
#if GGUF_SHUFFLE
// The shuffle prefill uses intel_sub_group_block_read on the plane-separated weight, which requires
// the sub-group width to equal SG_T. The dispatch launches local = {GGUF_GEMV_SG_SIZE=SG_T,1,1}.
__attribute__((intel_reqd_sub_group_size(SG_T)))
#endif
KERNEL(fc_gguf_transcode)(
    const __global uchar* W,        // GGUF block weights [N, K] (raw, or SG-shuffled if GGUF_SHUFFLE)
          __global uchar* WQ,       // out: packed low-bit weight [N, K] (u4 packed / u8) or f16
          __global half*  SC,       // out: per-group f16 scale [N, K/REQUANT_GROUP]
          __global uchar* ZP        // out: per-group u8 zero-point [N, K/REQUANT_GROUP] (unused for TRANSCODE_TO_F16)
)
{
    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    half blk_vals[GGUF_BLOCK_ELEM];
#if GGUF_SHUFFLE && (defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K) || defined(GGUF_IS_Q6_K) || \
                    defined(GGUF_IS_Q4_0) || defined(GGUF_IS_Q4_1) || defined(GGUF_IS_Q8_0))
    // ---- SG-shuffle prefill: one work-item owns one (n, blk); it gathers its lane's shuffled block
    //      bytes from the plane-separated SG layout and decodes them to 256 halfs (bit-exact). ----
    const int n   = (int)get_global_id(0);          // output row
    const int blk = (int)get_global_id(1);          // GGUF block index along K
    if (n >= N_SIZE || blk >= blocks_per_row)
        return;
    const int h   = n / SG_T;                        // row group (== sub-group; lane = get_sub_group_local_id())
    const uint entry_idx = (uint)h * (uint)blocks_per_row + (uint)blk;
#if defined(GGUF_IS_Q4_K)
    const uint off_pqs = 0u;
    const uint off_psl = (uint)N_SIZE * (uint)blocks_per_row * 128u;
    __private uint  pqs_u[32];   // 128 payload bytes, gathered as 32 uints via block reads
    __private uchar psl[16];
    FUNC_CALL(sh_gather_plane)(W, off_pqs + entry_idx * (uint)SG_T * 128u, pqs_u, 32);
    // psl SoA (per-lane, coalesced): sl@lid*4 ml@64+lid*4 sh@128+lid*2 mh@160+lid*2 d@192+lid*2 dmin@224+lid*2.
    // sl/ml are one uint per lane (block_read); sh/mh/d/dmin are one ushort per lane (block_read_us).
    {
        const __global uchar*  pe   = W + off_psl + entry_idx * (uint)SG_T * 16u;
        const __global uint*   pe_u = (const __global uint*)pe;
        const __global ushort* pe_s = (const __global ushort*)pe;
        const uint   sl = intel_sub_group_block_read   (pe_u + 0u);           // sl   @   0
        const uint   ml = intel_sub_group_block_read   (pe_u + (uint)SG_T);   // ml   @  64B (16 uints)
        const ushort sh = intel_sub_group_block_read_us(pe_s + 64u);          // sh   @ 128B (64 ushort)
        const ushort mh = intel_sub_group_block_read_us(pe_s + 80u);          // mh   @ 160B
        const ushort dd = intel_sub_group_block_read_us(pe_s + 96u);          // d    @ 192B
        const ushort dn = intel_sub_group_block_read_us(pe_s + 112u);         // dmin @ 224B
        psl[0]=(uchar)sl; psl[1]=(uchar)(sl>>8); psl[2]=(uchar)(sl>>16); psl[3]=(uchar)(sl>>24);
        psl[4]=(uchar)ml; psl[5]=(uchar)(ml>>8); psl[6]=(uchar)(ml>>16); psl[7]=(uchar)(ml>>24);
        psl[8]=(uchar)sh;  psl[9]=(uchar)(sh>>8);
        psl[10]=(uchar)mh; psl[11]=(uchar)(mh>>8);
        psl[12]=(uchar)dd; psl[13]=(uchar)(dd>>8);
        psl[14]=(uchar)dn; psl[15]=(uchar)(dn>>8);
    }
    FUNC_CALL(tq_decode_shuffle_q4k)((const __private uchar*)pqs_u, (const __private uchar*)psl, blk_vals);
#elif defined(GGUF_IS_Q5_K)
    const uint off_pqs = 0u;
    const uint off_pqh = (uint)N_SIZE * (uint)blocks_per_row * 128u;
    const uint off_psl = off_pqh + (uint)N_SIZE * (uint)blocks_per_row * 32u;
    __private uint  pqs_u[32];   // 128 low-nibble bytes, gathered as 32 uints via block reads
    __private uint  pqh_u[8];    // 32 high-bit bytes, gathered as 8 uints
    __private uchar psl[16];
    FUNC_CALL(sh_gather_plane)(W, off_pqs + entry_idx * (uint)SG_T * 128u, pqs_u, 32);
    FUNC_CALL(sh_gather_plane)(W, off_pqh + entry_idx * (uint)SG_T * 32u,  pqh_u, 8);
    // psl SoA (per-lane, coalesced): sl@lid*4 ml@64+lid*4 sh@128+lid*2 mh@160+lid*2 d@192+lid*2 dmin@224+lid*2.
    {
        const __global uchar*  pe   = W + off_psl + entry_idx * (uint)SG_T * 16u;
        const __global uint*   pe_u = (const __global uint*)pe;
        const __global ushort* pe_s = (const __global ushort*)pe;
        const uint   sl = intel_sub_group_block_read   (pe_u + 0u);           // sl   @   0
        const uint   ml = intel_sub_group_block_read   (pe_u + (uint)SG_T);   // ml   @  64B (16 uints)
        const ushort sh = intel_sub_group_block_read_us(pe_s + 64u);          // sh   @ 128B (64 ushort)
        const ushort mh = intel_sub_group_block_read_us(pe_s + 80u);          // mh   @ 160B
        const ushort dd = intel_sub_group_block_read_us(pe_s + 96u);          // d    @ 192B
        const ushort dn = intel_sub_group_block_read_us(pe_s + 112u);         // dmin @ 224B
        psl[0]=(uchar)sl; psl[1]=(uchar)(sl>>8); psl[2]=(uchar)(sl>>16); psl[3]=(uchar)(sl>>24);
        psl[4]=(uchar)ml; psl[5]=(uchar)(ml>>8); psl[6]=(uchar)(ml>>16); psl[7]=(uchar)(ml>>24);
        psl[8]=(uchar)sh;  psl[9]=(uchar)(sh>>8);
        psl[10]=(uchar)mh; psl[11]=(uchar)(mh>>8);
        psl[12]=(uchar)dd; psl[13]=(uchar)(dd>>8);
        psl[14]=(uchar)dn; psl[15]=(uchar)(dn>>8);
    }
    FUNC_CALL(tq_decode_shuffle_q5k)((const __private uchar*)pqs_u, (const __private uchar*)pqh_u, (const __private uchar*)psl, blk_vals);
#elif defined(GGUF_IS_Q6_K)
    const uint off_pql = 0u;
    const uint off_pqh = (uint)N_SIZE * (uint)blocks_per_row * 128u;
    const uint off_ps  = off_pqh + (uint)N_SIZE * (uint)blocks_per_row * 64u;
    const uint off_pd  = off_ps + (uint)N_SIZE * (uint)blocks_per_row * 16u;
    __private uint pql_u[32];    // 128 low-nibble bytes, gathered as 32 uints
    __private uint pqh_u[16];    // 64 high-bit bytes, gathered as 16 uints
    __private char ps[16];
    FUNC_CALL(sh_gather_plane)(W, off_pql + entry_idx * (uint)SG_T * 128u, pql_u, 32);
    FUNC_CALL(sh_gather_plane)(W, off_pqh + entry_idx * (uint)SG_T * 64u,  pqh_u, 16);
    {
        // ps: 16 scale bytes, scale si @ si*SG_T + lid -> one uchar block read (component si -> lane lid).
        const uchar16 scq = intel_sub_group_block_read_uc16((const __global uchar*)(W + off_ps + entry_idx * (uint)SG_T * 16u));
        ps[0]=as_char(scq.s0);  ps[1]=as_char(scq.s1);  ps[2]=as_char(scq.s2);  ps[3]=as_char(scq.s3);
        ps[4]=as_char(scq.s4);  ps[5]=as_char(scq.s5);  ps[6]=as_char(scq.s6);  ps[7]=as_char(scq.s7);
        ps[8]=as_char(scq.s8);  ps[9]=as_char(scq.s9);  ps[10]=as_char(scq.sa); ps[11]=as_char(scq.sb);
        ps[12]=as_char(scq.sc); ps[13]=as_char(scq.sd); ps[14]=as_char(scq.se); ps[15]=as_char(scq.sf);
        // pd: one f16 per lane @ lid*2 -> one ushort block read.
        const ushort dbits = intel_sub_group_block_read_us((const __global ushort*)(W + off_pd + entry_idx * (uint)SG_T * 2u));
        FUNC_CALL(tq_decode_shuffle_q6k)((const __private uchar*)pql_u, (const __private uchar*)pqh_u, (const __private char*)ps, as_half(dbits), blk_vals);
    }
#elif defined(GGUF_IS_Q4_0)
    // Q4_0 super-block: pqs[128] (32 uints) + pd[8 fp16].
    const uint off_pqs = 0u;
    const uint off_pd  = (uint)N_SIZE * (uint)blocks_per_row * 128u;
    __private uint pqs_u[32];
    __private half dv[8];
    FUNC_CALL(sh_gather_plane)(W, off_pqs + entry_idx * (uint)SG_T * 128u, pqs_u, 32);
    {
        const __global ushort* pd_s = (const __global ushort*)(W + off_pd);
        const ushort8 d8 = intel_sub_group_block_read_us8(pd_s + entry_idx * (uint)SG_T * 8u);
        dv[0]=as_half(d8.s0); dv[1]=as_half(d8.s1); dv[2]=as_half(d8.s2); dv[3]=as_half(d8.s3);
        dv[4]=as_half(d8.s4); dv[5]=as_half(d8.s5); dv[6]=as_half(d8.s6); dv[7]=as_half(d8.s7);
    }
    FUNC_CALL(tq_decode_shuffle_q4_0)((const __private uchar*)pqs_u, (const __private half*)dv, blk_vals);
#elif defined(GGUF_IS_Q4_1)
    // Q4_1 super-block: pqs[128] (32 uints) + pd[8 fp16] + pm[8 fp16].
    const uint off_pqs = 0u;
    const uint off_pd  = (uint)N_SIZE * (uint)blocks_per_row * 128u;
    const uint off_pm  = off_pd + (uint)N_SIZE * (uint)blocks_per_row * 16u;
    __private uint pqs_u[32];
    __private half dv[8];
    __private half mv[8];
    FUNC_CALL(sh_gather_plane)(W, off_pqs + entry_idx * (uint)SG_T * 128u, pqs_u, 32);
    {
        const __global ushort* pd_s = (const __global ushort*)(W + off_pd);
        const __global ushort* pm_s = (const __global ushort*)(W + off_pm);
        const ushort8 d8 = intel_sub_group_block_read_us8(pd_s + entry_idx * (uint)SG_T * 8u);
        const ushort8 m8 = intel_sub_group_block_read_us8(pm_s + entry_idx * (uint)SG_T * 8u);
        dv[0]=as_half(d8.s0); dv[1]=as_half(d8.s1); dv[2]=as_half(d8.s2); dv[3]=as_half(d8.s3);
        dv[4]=as_half(d8.s4); dv[5]=as_half(d8.s5); dv[6]=as_half(d8.s6); dv[7]=as_half(d8.s7);
        mv[0]=as_half(m8.s0); mv[1]=as_half(m8.s1); mv[2]=as_half(m8.s2); mv[3]=as_half(m8.s3);
        mv[4]=as_half(m8.s4); mv[5]=as_half(m8.s5); mv[6]=as_half(m8.s6); mv[7]=as_half(m8.s7);
    }
    FUNC_CALL(tq_decode_shuffle_q4_1)((const __private uchar*)pqs_u, (const __private half*)dv, (const __private half*)mv, blk_vals);
#else  // GGUF_IS_Q8_0
    // Q8_0 super-block: pqs[256] (64 uints) + pd[8 fp16].
    const uint off_pqs = 0u;
    const uint off_pd  = (uint)N_SIZE * (uint)blocks_per_row * 256u;
    __private uint pqs_u[64];
    __private half dv[8];
    FUNC_CALL(sh_gather_plane)(W, off_pqs + entry_idx * (uint)SG_T * 256u, pqs_u, 64);
    {
        const __global ushort* pd_s = (const __global ushort*)(W + off_pd);
        const ushort8 d8 = intel_sub_group_block_read_us8(pd_s + entry_idx * (uint)SG_T * 8u);
        dv[0]=as_half(d8.s0); dv[1]=as_half(d8.s1); dv[2]=as_half(d8.s2); dv[3]=as_half(d8.s3);
        dv[4]=as_half(d8.s4); dv[5]=as_half(d8.s5); dv[6]=as_half(d8.s6); dv[7]=as_half(d8.s7);
    }
    FUNC_CALL(tq_decode_shuffle_q8_0)((const __private char*)pqs_u, (const __private half*)dv, blk_vals);
#endif
#else
    const int n   = (int)get_global_id(0);          // output row (subgroup lane axis, padded to SG)
    const int blk = (int)get_global_id(1);          // GGUF block index along K
    if (n >= N_SIZE || blk >= blocks_per_row)
        return;
    // Decode the whole GGUF block ONCE. Every REQUANT group inside it reuses this decoded window, so
    // the expensive bit-unpacking runs a single time per block instead of once per group.
    const __global uchar* w_row = W + (uint)n * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
    FUNC_CALL(tq_decode_block)(w_row + (uint)blk * GGUF_BLOCK_BYTES, blk_vals);
#endif

    const uint row_base = (uint)n * (uint)K_SIZE;
#if TRANSCODE_TO_F16
    // f16 prefill (OV_GPU_GGUF_PREFILL_F16=1): write the decoded block straight to the f16 weight
    // scratchpad. No per-group requant, no scale, no zero-point. Weight md [K,N] `ba` -> physical
    // [N,K] f16: element (n,k) at n*K + k. SC and ZP are left untouched.
    __global half* wq_f16 = (__global half*)WQ;
    const uint k_base   = (uint)blk * (uint)GGUF_BLOCK_ELEM;
    const uint f16_base = row_base + k_base;
    // Per-lane wide store: 256 scalar half stores -> 32 half8 stores, i.e. 8x fewer store messages.
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < GGUF_BLOCK_ELEM; i += 8) {
        const half8 v = (half8)(blk_vals[i + 0], blk_vals[i + 1], blk_vals[i + 2], blk_vals[i + 3],
                                blk_vals[i + 4], blk_vals[i + 5], blk_vals[i + 6], blk_vals[i + 7]);
        vstore8(v, 0, wq_f16 + f16_base + (uint)i);
    }
#else
    const int groups_per_block = GGUF_BLOCK_ELEM / REQUANT_GROUP;

    // Per-group requantization for each REQUANT group within the decoded block.
    // TRANSCODE_ASYMMETRIC=1 (u4/u8 unsigned + u8 ZP): Q4_K, Q4_0, Q4_1, Q3_K, IQ3_*.
    //   val ≈ (q - zp) * scale,  q ∈ [0, QMAX]  — matches NNCF FP16-4BIT / jit:gemm:any u4/u8
    // TRANSCODE_ASYMMETRIC=0 (s8 signed symmetric, no ZP): Q5_K, Q6_K, IQ2_*.
    //   val ≈ q * scale,  q ∈ [-QMAX, QMAX]  — jit:gemm:any dy_quant_enabled s8×s8
    //
    // Performance: the reduction and quantize inner loops are 8-wide vectorized (REQUANT_GROUP=32
    // -> 4 vector ops instead of 32 scalar) and use hardware `convert_<t><n>_sat_rte` which fuses
    // round-to-nearest-even + saturating clamp + narrowing convert into ONE instruction per lane,
    // replacing the old round()+clamp()+cast chain (3 ops -> 1). min/max reductions run in half
    // (native 2x rate, exact for comparisons); scale/inv_scale are computed in float.
#if (REQUANT_GROUP % 8) != 0
#error "fc_gguf_transcode: REQUANT_GROUP must be a multiple of 8 for the vectorized requant path"
#endif
    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int off_in_blk = gi * REQUANT_GROUP;
        const int g  = blk * groups_per_block + gi;
        const int k0 = g * REQUANT_GROUP;
        const uint group_off = (uint)g * (uint)N_SIZE + (uint)n;
        const __private half* grp = blk_vals + off_in_blk;

#if TRANSCODE_ASYMMETRIC
        // ---- Asymmetric: vectorized min/max (half, exact), unsigned output [0..QMAX] + u8 ZP ----
        half8 vmx = vload8(0, grp);
        half8 vmn = vmx;
        __attribute__((opencl_unroll_hint))
        for (int i = 8; i < REQUANT_GROUP; i += 8) {
            const half8 v = vload8(0, grp + i);
            vmx = fmax(vmx, v);
            vmn = fmin(vmn, v);
        }
        const half4 mx4 = fmax(vmx.lo, vmx.hi);
        const half4 mn4 = fmin(vmn.lo, vmn.hi);
        const half2 mx2 = fmax(mx4.lo, mx4.hi);
        const half2 mn2 = fmin(mn4.lo, mn4.hi);
        const float vmax = (float)fmax(mx2.s0, mx2.s1);
        const float vmin = (float)fmin(mn2.s0, mn2.s1);

        const float range     = vmax - vmin;
        const float scale     = (range > 0.0f) ? (range * (1.0f / (float)QMAX)) : 1.0f;
        const float inv_scale = (range > 0.0f) ? ((float)QMAX / range) : 0.0f;
        const float zp_f      = -vmin * inv_scale;
        SC[group_off] = (half)scale;
        // zp_f ∈ [0, QMAX] by construction; sat-convert to u8 then clamp to QMAX (u4: 15, u8: 255).
        ZP[group_off] = min(convert_uchar_sat_rte(zp_f), (uchar)QMAX);

#  if TRANSCODE_TO_I4
        // u4: quantize 8 at a time (float8 FMA + fused sat-rte convert), pack even|odd<<4 into 4 bytes.
        uchar u4buf[REQUANT_GROUP / 2];
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 8) {
            const float8 v  = convert_float8(vload8(0, grp + i));
            const float8 qf = v * inv_scale + zp_f;                    // = (v - vmin) * inv_scale
            const uchar8 q  = min(convert_uchar8_sat_rte(qf), (uchar8)QMAX);  // round+clamp[0,QMAX]
            // byte j holds element 2j (low nibble) | element 2j+1 (high nibble).
            const uchar4 packed = q.even | (uchar4)(q.odd << (uchar4)4);
            vstore4(packed, 0, u4buf + (i >> 1));
        }
        // Flush u4buf (REQUANT_GROUP/2 bytes) in 16-byte chunks; unrolled at compile time.
        const uint byte_base = (row_base + (uint)k0) >> 1;
        __attribute__((opencl_unroll_hint))
        for (int bi = 0; bi < REQUANT_GROUP / 2; bi += 16)
            vstore16(vload16(0, u4buf + bi), 0, WQ + byte_base + bi);
#  else
        // u8: quantize 8 at a time; convert_uchar8_sat_rte rounds + clamps to [0, 255] = [0, QMAX].
        __global uchar* dst = WQ + row_base + (uint)k0;
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 8) {
            const float8 v  = convert_float8(vload8(0, grp + i));
            const float8 qf = v * inv_scale + zp_f;
            const uchar8 q  = convert_uchar8_sat_rte(qf);
            vstore8(q, 0, dst + i);
        }
#  endif  // TRANSCODE_TO_I4

#else  // TRANSCODE_ASYMMETRIC=0: symmetric s8
        // ---- Symmetric: vectorized max(|v|) (half), signed s8 output [-QMAX..QMAX], ZP not written ----
        // For Q5_K/Q6_K: jit:gemm:any dy_quant_enabled s8×s8 W4A8 on Xe2/B580.
        half8 amx = fabs(vload8(0, grp));
        __attribute__((opencl_unroll_hint))
        for (int i = 8; i < REQUANT_GROUP; i += 8) {
            amx = fmax(amx, fabs(vload8(0, grp + i)));
        }
        const half4 a4 = fmax(amx.lo, amx.hi);
        const half2 a2 = fmax(a4.lo, a4.hi);
        const float amax = (float)fmax(a2.s0, a2.s1);

        const float scale     = (amax > 0.0f) ? (amax * (1.0f / (float)QMAX)) : 1.0f;
        const float inv_scale = (amax > 0.0f) ? ((float)QMAX / amax) : 0.0f;
        SC[group_off] = (half)scale;
        // ZP not written (symmetric: zero-point = 0)

        // s8: quantize 8 at a time; convert_char8_sat_rte rounds + clamps to [-128,127].
        // With amax scaling, qf ∈ [-QMAX, QMAX] = [-127, 127], so -128 never occurs (matches [-QMAX,QMAX]).
        __global char* dst_s8 = (__global char*)WQ + row_base + (uint)k0;
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 8) {
            const float8 v  = convert_float8(vload8(0, grp + i));
            const float8 qf = v * inv_scale;
            const char8  q  = convert_char8_sat_rte(qf);
            vstore8(q, 0, dst_s8 + i);
        }
#endif  // TRANSCODE_ASYMMETRIC
    }
#endif  // TRANSCODE_TO_F16
}
