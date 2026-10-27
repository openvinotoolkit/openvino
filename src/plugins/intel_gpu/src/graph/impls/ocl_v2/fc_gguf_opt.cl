// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF FullyConnected kernel.
//
// Computes  C[bm, n] = sum_k  A[bm, k] * W[n, k]
// where W is a raw GGUF block-quantised weight matrix [N, K] consumed directly from HBM and
// dequantised in registers (never materialised to f16/f32 in memory). The activation A is f16/f32
// [BM, K] and the output C is f16/f32 [BM, N]. BM is the flattened batch*sequence dimension, so the
// same kernel serves both the M==1 decode (GEMV) and M>1 prefill (GEMM) phases.
//
// One subgroup (SG_SIZE lanes) cooperatively owns one (n, bm) output: the blocks of W's row `n` are
// striped across the lanes (lane L decodes blocks L, L+SG_SIZE, ...), each lane streams its block's
// dot product against the matching A slice WITHOUT materialising the dequantised block in private
// memory, and a single sub_group_reduce_add collapses the partials. Striping keeps the SG_SIZE
// lanes sweeping a contiguous block window each step (coalesced) and keeps all SIMD lanes busy --
// the previous 1-work-item-per-output layout left 15/16 lanes idle (LWS=1) and was memory-starved.
// The streaming dot mirrors the canonical ggml reference (ggml-quants.c) and the CPU reference in
// the GGUF frontend (src/frontends/gguf/src/builders/dequantize.cpp).
//
// The packed GGUF source format is selected at JIT time by exactly one GGUF_IS_<TYPE> flag, together
// with GGUF_BLOCK_ELEM (logical elements per block) and GGUF_BLOCK_BYTES (bytes per block).
//
// Helper functions are wrapped in FUNC()/FUNC_CALL() so their names are decorated with the kernel
// entry point — multiple GGUF FC kernels (different shapes/formats) are batch-compiled into a single
// OpenCL program, and undecorated names would collide ("redefinition of ...").

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "gguf/gguf_iq_tables.hpp"

// Reconstruct a half from two little-endian bytes (GGUF is little-endian, as is every OV host/target).
inline half FUNC(gguf_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

inline float FUNC(gguf_load_activation)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* A,
                                        const uint bm,
                                        const uint k) {
    const uint b = bm / INPUT0_FEATURE_NUM;
    const uint f = bm - b * INPUT0_FEATURE_NUM;
    return (float)A[INPUT0_GET_INDEX(b, f, k, 0)];
}

inline uint FUNC(gguf_output_index)(OPTIONAL_SHAPE_INFO_ARG const uint bm, const uint n) {
    const uint b = bm / OUTPUT_FEATURE_NUM;
    const uint f = bm - b * OUTPUT_FEATURE_NUM;
    return OUTPUT_GET_INDEX(b, f, n, 0);
}

// ============================================================================
// Per-format streaming block dot. Each returns sum_{j in [0, GGUF_BLOCK_ELEM)} a[j] * dequant(blk[j])
// for the block starting at `blk` against the activation slice `a`, accumulating in float without
// materialising the dequantised block (keeps register pressure low so SG_SIZE lanes stay resident).
// ============================================================================

#if defined(GGUF_IS_Q4_0)
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const int lo = (int)(qs[j] & 0x0F) - 8;
        const int hi = (int)(qs[j] >> 4) - 8;
        acc += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)j) * ((float)lo * d);
        acc += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)j + 16) * ((float)hi * d);
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q4_1)
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float m = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* qs = blk + 4;
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const float lo = (float)(qs[j] & 0x0F);
        const float hi = (float)(qs[j] >> 4);
        acc += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)j)      * (lo * d + m);
        acc += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)j + 16) * (hi * d + m);
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q8_0)
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    float acc = 0.0f;
    for (int j = 0; j < 32; ++j) {
        acc += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)j) * ((float)qs[j] * d);
    }
    return acc;
}
#endif

// 6-bit packed sub-block scale/min extraction shared by Q4_K / Q5_K (ggml get_scale_min_k4).
#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
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

#if defined(GGUF_IS_Q4_K)
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d    = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;    // 12 bytes
    const __global uchar* qs     = blk + 16;   // 128 bytes
    float acc = 0.0f;
    int ai = 0;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = dmin * m;
        FUNC_CALL(gguf_get_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = dmin * m;
        // Factor the per-element (d*q - m) into d*sum(a*q) - m*sum(a): one fma + one add per element.
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float av = FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(ai + l));
            sq1 += av * (float)(qs[l] & 0x0F);
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(ai + 32 + l));
            sq2 += av * (float)(qs[l] >> 4);
            sa2 += av;
        }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        qs += 32;
        is += 2;
        ai += 64;
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q5_K)
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d    = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;    // 12 bytes
    const __global uchar* qh     = blk + 16;   // 32 bytes (high bit-plane)
    const __global uchar* ql     = blk + 48;   // 128 bytes (low 4 bits)
    float acc = 0.0f;
    int ai = 0;
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
        // Factor the per-element (d*q - m) into d*sum(a*q) - m*sum(a): one fma + one add per element.
        float sq1 = 0.0f, sa1 = 0.0f, sq2 = 0.0f, sa2 = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float av = FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(ai + l));
            const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            sq1 += av * (float)q;
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(ai + 32 + l));
            const int q = (int)(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            sq2 += av * (float)q;
            sa2 += av;
        }
        acc += d1 * sq1 - m1 * sa1 + d2 * sq2 - m2 * sa2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        ai += 64;
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q6_K)
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const __global uchar* ql = blk;            // 128 bytes (low 4 bits)
    const __global uchar* qh = blk + 128;      // 64 bytes (high 2 bits)
    const __global char*  sc = (const __global char*)(blk + 192);  // 16 signed scales
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk + 208);
    // Four independent accumulators + inner unroll: Q6_K decode is latency-bound on the four
    // length-32 dependent FMA chains, so unrolling lets the (independent) per-element unpacks
    // pipeline and the four chains overlap. Measured +27% (24.9% -> 31.7% of B580 BW roofline) vs
    // the single-accumulator form. Q5_K showed the opposite (register/occupancy-bound) so only Q6_K
    // uses this form -- the split is intentionally format-local.
    float acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        __attribute__((opencl_unroll_hint(8)))
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(o + l + 0))  * (d * (float)sc[is + 0] * q1);
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(o + l + 32)) * (d * (float)sc[is + 2] * q2);
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(o + l + 64)) * (d * (float)sc[is + 4] * q3);
            acc4 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + (uint)(o + l + 96)) * (d * (float)sc[is + 6] * q4);
        }
        o  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc1 + acc2 + acc3 + acc4;
}
#endif

#if defined(GGUF_IS_Q3_K)
// Q3_K block layout (110 bytes / 256 elements; ggml block_q3_k):
//   hmask:         blk[0..32]    (32 bytes high-mask planes)
//   qs:            blk[32..96]   (64 bytes, two 128-elem halves; 2-bit payload in 4 bit-planes)
//   packed_scales: blk[96..108]  (12 bytes, repacked into 16 signed per-16 scales)
//   d_all:         blk[108..110] (ggml_half global scale)
//
// Canonical unpack mirrors frontend dequantize_q3_k():
//   q = ((qs >> shift) & 0x3) - ((hmask & mask) ? 0 : 4), q in {-4..3}
//   w = d_all * (scale_i - 32) * q
//
// Optimisations vs. the literal CPU port (same final value, FP-reorder only):
//   * 32B hmask + 64B qs are pulled into private memory ONCE -- the naive loop re-reads each
//     qs byte 4x and each hmask byte 8x from __global per block (~384 redundant uchar loads).
//   * The 12-byte packed scales are decoded into sixteen pre-scaled sub-scales (dl[0..15])
//     up front, replacing the 4-way ternary on `is` plus the byte-shift on a uint.
//   * Four parallel accumulators (one per j-shift) -- same idiom as the Q6_K decode above --
//     break the 256-FMA serial chain into four 64-FMA chains so the independent per-lane
//     unpacks pipeline across the four accumulators. Q3_K and Q6_K have the same op-mix profile
//     (low-bit unpack + per-lane FMA), so the same 4-way structure applies.
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    // 1. Cache global block bytes once.
    __private uchar hmask[32];
    __private uchar qs[64];
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 32; ++i) hmask[i] = blk[i];           // hmask = blk[0..31]
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 64; ++i) qs[i]    = blk[32 + i];      // qs    = blk[32..95]

    const float d_all = (float)FUNC_CALL(gguf_load_f16)(blk + 108);

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

    // 3. Four parallel accumulators (one per j-shift), folded at end. The inner lane loop
    //    interleaves four independent FMAs so they pipeline across the four accumulator chains.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int nh = 0; nh < 2; ++nh) {
        const uint  ai_h   = 128u * (uint)nh;
        const uchar mask0  = (uchar)(1u << (4 * nh + 0));
        const uchar mask1  = (uchar)(1u << (4 * nh + 1));
        const uchar mask2  = (uchar)(1u << (4 * nh + 2));
        const uchar mask3  = (uchar)(1u << (4 * nh + 3));
        const float dl0_lo = dl[8 * nh + 0];
        const float dl0_hi = dl[8 * nh + 1];
        const float dl1_lo = dl[8 * nh + 2];
        const float dl1_hi = dl[8 * nh + 3];
        const float dl2_lo = dl[8 * nh + 4];
        const float dl2_hi = dl[8 * nh + 5];
        const float dl3_lo = dl[8 * nh + 6];
        const float dl3_hi = dl[8 * nh + 7];
        const __private uchar* qs_h = qs + 32 * nh;

        // Lo lanes (l in 0..15) of all four sub-blocks (j=0..3), in parallel.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 16; ++l) {
            const uchar qsl = qs_h[l];
            const uchar hml = hmask[l];
            const int q0 = (int)((qsl >> 0) & 3) - ((hml & mask0) ? 0 : 4);
            const int q1 = (int)((qsl >> 2) & 3) - ((hml & mask1) ? 0 : 4);
            const int q2 = (int)((qsl >> 4) & 3) - ((hml & mask2) ? 0 : 4);
            const int q3 = (int)((qsl >> 6) & 3) - ((hml & mask3) ? 0 : 4);
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h +   0u + (uint)l) * (dl0_lo * (float)q0);
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h +  32u + (uint)l) * (dl1_lo * (float)q1);
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h +  64u + (uint)l) * (dl2_lo * (float)q2);
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h +  96u + (uint)l) * (dl3_lo * (float)q3);
        }
        // Hi lanes (qs_h[l+16], hmask[l+16]) of all four sub-blocks, in parallel.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 16; ++l) {
            const uchar qsl = qs_h[l + 16];
            const uchar hml = hmask[l + 16];
            const int q0 = (int)((qsl >> 0) & 3) - ((hml & mask0) ? 0 : 4);
            const int q1 = (int)((qsl >> 2) & 3) - ((hml & mask1) ? 0 : 4);
            const int q2 = (int)((qsl >> 4) & 3) - ((hml & mask2) ? 0 : 4);
            const int q3 = (int)((qsl >> 6) & 3) - ((hml & mask3) ? 0 : 4);
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h +  16u + (uint)l) * (dl0_hi * (float)q0);
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h +  48u + (uint)l) * (dl1_hi * (float)q1);
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h +  80u + (uint)l) * (dl2_hi * (float)q2);
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai_h + 112u + (uint)l) * (dl3_hi * (float)q3);
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}
#endif

#if defined(GGUF_IS_IQ3_XXS)
// IQ3_XXS streaming dot. Optimisations vs the literal CPU port (same final value, FP-reorder only):
//   * Four parallel accumulators round-robin over the 8 grid bytes per l-iter -- breaks the
//     256-FMA serial chain into four 64-FMA chains (same idiom as the Q6_K decode above; the
//     same trick gave Q6_K +27% on B580). The per-iter unpack work (grid lookups, sign LUT,
//     uchar byte extracts) is identical, so the four chains pipeline cleanly.
//   * Inner l-loop hinted to fully unroll so `7*l`, `2*l`, and sign-bit masks collapse to
//     compile-time constants -> the 8 grid lookups + sign extracts schedule together.
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    const __global uchar* scales_signs = blk + 2 + 64;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const __global uchar* p4 = scales_signs + 4 * ib32;
        const uint aux32 = (uint)p4[0] | ((uint)p4[1] << 8) | ((uint)p4[2] << 16) | ((uint)p4[3] << 24);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const __global uchar* qsp = qs + 8 * ib32;
        const uint ai_b = 32u * (uint)ib32;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 4; ++l) {
            const uchar signs = CONST_ARRAY_REF(ksigns_iq2xs)[(aux32 >> (7 * l)) & 127u];
            const uint  g1    = CONST_ARRAY_REF(iq3xxs_grid)[qsp[2 * l + 0]];
            const uint  g2    = CONST_ARRAY_REF(iq3xxs_grid)[qsp[2 * l + 1]];
            const uint  ai    = ai_b + 8u * (uint)l;
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 0u) * (db * (float)(uchar)( g1        & 0xFFu) * ((signs & 1u  ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 1u) * (db * (float)(uchar)((g1 >>  8) & 0xFFu) * ((signs & 2u  ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 2u) * (db * (float)(uchar)((g1 >> 16) & 0xFFu) * ((signs & 4u  ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 3u) * (db * (float)(uchar)((g1 >> 24) & 0xFFu) * ((signs & 8u  ) ? -1.0f : 1.0f));
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 4u) * (db * (float)(uchar)( g2        & 0xFFu) * ((signs & 16u ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 5u) * (db * (float)(uchar)((g2 >>  8) & 0xFFu) * ((signs & 32u ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 6u) * (db * (float)(uchar)((g2 >> 16) & 0xFFu) * ((signs & 64u ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 7u) * (db * (float)(uchar)((g2 >> 24) & 0xFFu) * ((signs & 128u) ? -1.0f : 1.0f));
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}
#endif

#if defined(GGUF_IS_IQ3_S)
// IQ3_S block layout (110 bytes / 256 elements; ggml block_iq3_s):
//   d:      blk[0..2]    (ggml_half scale)
//   qs:     blk[2..66]   (64 bytes, low 8 bits of 64 grid indices, 8 per ib32 sub-block)
//   qh:     blk[66..74]  (8 bytes, 9th bit (MSB) of every index, 1 byte per ib32)
//   signs:  blk[74..106] (32 bytes, 4 sign bytes per ib32; bit j masks element j of an 8-wide quartet)
//   scales: blk[106..110](4 bytes; 8 4-bit per-ib32 scales packed two-per-byte)
// Per-ib32 sub-scale db = d * (1 + 2 * scale_nibble); each ib32 owns 32 logical elements.
// Grid is 512-entry (9-bit index) vs IQ3_XXS's 256-entry; sign byte is used directly (no ksigns LUT).

// Optimisations vs the literal CPU port (same final value, FP-reorder only):
//   * Four parallel accumulators round-robin over the 8 grid bytes per l-iter (Q6_K idiom).
//   * Inner l-loop hinted to fully unroll so `2*l` shifts and qsp[2*l] indices collapse to
//     compile-time constants for the 9-bit index assembly.
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* qh     = blk + 66;
    const __global uchar* signs  = blk + 74;
    const __global uchar* scales = blk + 106;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc_byte = scales[ib32 >> 1];
        const int   sc4     = (ib32 & 1) ? (int)(sc_byte >> 4) : (int)(sc_byte & 0xF);
        const float db      = d * (float)(1 + 2 * sc4);
        const uchar qhb     = qh[ib32];
        const __global uchar* qsp = qs + 8 * ib32;
        const __global uchar* sgp = signs + 4 * ib32;
        const uint ai_b = 32u * (uint)ib32;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 4; ++l) {
            const uint  idx1 = (uint)qsp[2*l + 0] | ((((uint)qhb >> (2*l    )) & 1u) << 8);
            const uint  idx2 = (uint)qsp[2*l + 1] | ((((uint)qhb >> (2*l + 1)) & 1u) << 8);
            const uint  g1   = CONST_ARRAY_REF(iq3s_grid)[idx1];
            const uint  g2   = CONST_ARRAY_REF(iq3s_grid)[idx2];
            const uchar sb   = sgp[l];
            const uint  ai   = ai_b + 8u * (uint)l;
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 0u) * (db * (float)(uchar)( g1        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 1u) * (db * (float)(uchar)((g1 >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 2u) * (db * (float)(uchar)((g1 >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 3u) * (db * (float)(uchar)((g1 >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 4u) * (db * (float)(uchar)( g2        & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 5u) * (db * (float)(uchar)((g2 >>  8) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 6u) * (db * (float)(uchar)((g2 >> 16) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 7u) * (db * (float)(uchar)((g2 >> 24) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}
#endif

#if defined(GGUF_IS_IQ2_S)
// IQ2_S block layout (82 bytes / 256 elements; ggml block_iq2_s):
//   d:      blk[0..2]    (ggml_half super-scale)
//   qs:     blk[2..34]   (32 bytes; 4 per ib32; low 8 bits of a 10-bit grid index)
//   signs:  blk[34..66]  (32 bytes; 4 per ib32; bit j masks element j of the 8-wide grid word)
//   qh:     blk[66..74]  (8 bytes; 1 per ib32; carries the 2 high bits of each of the 4 indices,
//                          extracted as `(qh[ib32] << (8 - 2*l)) & 0x300`)
//   scales: blk[74..82]  (8 bytes; 1 per ib32; low nibble => db0 for l=0,1 (elems 0..15),
//                          high nibble => db1 for l=2,3 (elems 16..31)); db = d*(0.5+nibble)*0.25.
// Grid is 1024-entry uint64 (8 packed unsigned magnitudes per row; values in {8, 25, 43}); access
// it as `ulong` (matches CONST_ARRAY_DECL = __constant size_t). Sign byte is consumed directly --
// no ksigns_iq2xs LUT here (the IQ3_XXS-style aux32-sign packing does not apply).
// Optimisations vs the literal CPU port (same final value, FP-reorder only):
//   * Four parallel accumulators round-robin over the 8 grid bytes per l-iter (Q6_K idiom).
//   * The l-loop is peeled into l=0,1 (use db0) and l=2,3 (use db1) so the per-element
//     `(l<2) ? db0 : db1` ternary becomes a compile-time constant inside each peeled body.
//   * Both peeled halves are fully unrolled so qsp[l]/sgp[l]/(qhb<<(8-2*l)) folds away.
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* signs  = blk + 34;
    const __global uchar* qh     = blk + 66;
    const __global uchar* scales = blk + 74;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc  = scales[ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;  // elems 0..15 (l = 0, 1)
        const float db1 = d * (0.5f + (float)(sc >> 4))  * 0.25f;  // elems 16..31 (l = 2, 3)
        const uchar qhb = qh[ib32];
        const __global uchar* qsp = qs    + 4 * ib32;
        const __global uchar* sgp = signs + 4 * ib32;
        const uint ai_b = 32u * (uint)ib32;
        // l = 0, 1 -> elems 0..15 use db0.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 2; ++l) {
            const uint  idx = (uint)qsp[l] | (((uint)qhb << (8 - 2 * l)) & 0x300u);
            const ulong g   = CONST_ARRAY_REF(iq2s_grid)[idx];
            const uchar sb  = sgp[l];
            const uint  ai  = ai_b + 8u * (uint)l;
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 0u) * (db0 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 1u) * (db0 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 2u) * (db0 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 3u) * (db0 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 4u) * (db0 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 5u) * (db0 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 6u) * (db0 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 7u) * (db0 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
        // l = 2, 3 -> elems 16..31 use db1.
        __attribute__((opencl_unroll_hint))
        for (int l = 2; l < 4; ++l) {
            const uint  idx = (uint)qsp[l] | (((uint)qhb << (8 - 2 * l)) & 0x300u);
            const ulong g   = CONST_ARRAY_REF(iq2s_grid)[idx];
            const uchar sb  = sgp[l];
            const uint  ai  = ai_b + 8u * (uint)l;
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 0u) * (db1 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 1u) * (db1 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 2u) * (db1 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 3u) * (db1 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 4u) * (db1 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 5u) * (db1 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 6u) * (db1 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 7u) * (db1 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}
#endif

#if defined(GGUF_IS_IQ2_XS)
// IQ2_XS block layout (74 bytes / 256 elements; ggml block_iq2_xs):
//   d:      blk[0..2]    (ggml_half super-scale)
//   qs:     blk[2..66]   (32 packed uint16; 4 per ib32, 8 bytes per ib32). Each uint16:
//                          bits 0..8  (9 bits)  -> iq2xs_grid index (0..511)
//                          bits 9..15 (7 bits)  -> ksigns_iq2xs LUT index (0..127) -> sign byte mask
//   scales: blk[66..74]  (8 bytes; 1 per ib32; low nibble => db0 for l=0,1 (elems 0..15),
//                          high nibble => db1 for l=2,3 (elems 16..31)); db = d*(0.5+nibble)*0.25.
// Grid is 512-entry uint64 (8 packed unsigned magnitudes per row; values in {8, 25, 43}); access
// it as `ulong` (matches CONST_ARRAY_DECL = __constant size_t). Sign byte comes from the shared
// 7-bit ksigns_iq2xs LUT (same path as IQ3_XXS / IQ2_XXS) -- different from IQ2_S which stores a
// plain per-byte sign mask. Dual sub-scale (db0/db1) mirrors IQ2_S; see transcode_target() doc.
// Optimisations: same as IQ2_S above (4-way acc + peeled l-loop + unroll hints).
inline float FUNC(gguf_block_dot)(OPTIONAL_SHAPE_INFO_ARG const __global uchar* blk,
                                  const __global INPUT0_TYPE* A,
                                  const uint bm,
                                  const uint k0) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs     = blk + 2;
    const __global uchar* scales = blk + 66;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uchar sc  = scales[ib32];
        const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;  // elems 0..15 (l = 0, 1)
        const float db1 = d * (0.5f + (float)(sc >> 4))  * 0.25f;  // elems 16..31 (l = 2, 3)
        const __global uchar* qsp = qs + 8 * ib32;                 // 4 packed uint16 = 8 bytes per ib32
        const uint ai_b = 32u * (uint)ib32;
        // l = 0, 1 -> elems 0..15 use db0.
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 2; ++l) {
            const uint  q   = (uint)qsp[2 * l] | ((uint)qsp[2 * l + 1] << 8);
            const ulong g   = CONST_ARRAY_REF(iq2xs_grid)[q & 0x1FFu];
            const uchar sb  = CONST_ARRAY_REF(ksigns_iq2xs)[(q >> 9) & 0x7Fu];
            const uint  ai  = ai_b + 8u * (uint)l;
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 0u) * (db0 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 1u) * (db0 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 2u) * (db0 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 3u) * (db0 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 4u) * (db0 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 5u) * (db0 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 6u) * (db0 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 7u) * (db0 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
        // l = 2, 3 -> elems 16..31 use db1.
        __attribute__((opencl_unroll_hint))
        for (int l = 2; l < 4; ++l) {
            const uint  q   = (uint)qsp[2 * l] | ((uint)qsp[2 * l + 1] << 8);
            const ulong g   = CONST_ARRAY_REF(iq2xs_grid)[q & 0x1FFu];
            const uchar sb  = CONST_ARRAY_REF(ksigns_iq2xs)[(q >> 9) & 0x7Fu];
            const uint  ai  = ai_b + 8u * (uint)l;
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 0u) * (db1 * (float)(uchar)( g        & 0xFFu) * ((sb & 1u  ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 1u) * (db1 * (float)(uchar)((g >>  8) & 0xFFu) * ((sb & 2u  ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 2u) * (db1 * (float)(uchar)((g >> 16) & 0xFFu) * ((sb & 4u  ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 3u) * (db1 * (float)(uchar)((g >> 24) & 0xFFu) * ((sb & 8u  ) ? -1.0f : 1.0f));
            acc0 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 4u) * (db1 * (float)(uchar)((g >> 32) & 0xFFu) * ((sb & 16u ) ? -1.0f : 1.0f));
            acc1 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 5u) * (db1 * (float)(uchar)((g >> 40) & 0xFFu) * ((sb & 32u ) ? -1.0f : 1.0f));
            acc2 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 6u) * (db1 * (float)(uchar)((g >> 48) & 0xFFu) * ((sb & 64u ) ? -1.0f : 1.0f));
            acc3 += FUNC_CALL(gguf_load_activation)(OPTIONAL_SHAPE_INFO_TENSOR A, bm, k0 + ai + 7u) * (db1 * (float)(uchar)((g >> 56) & 0xFFu) * ((sb & 128u) ? -1.0f : 1.0f));
        }
    }
    return acc0 + acc1 + acc2 + acc3;
}
#endif

// NROW = number of output rows one subgroup owns (register/multi-row blocking). For the M=1 decode
// GEMV every output row re-reads the SAME activation slice A[bm, :]; owning NROW rows lets the kernel
// stripe each K-block once and reuse the (L2-resident) activation across the NROW per-row dot products,
// amortising the per-row loop/decode overhead that throttles this scalar-float path. Default 1 keeps
// the original single-output behaviour. Layout-invariant: only which rows a subgroup owns changes, not
// the (N,K) weight byte layout. The win is shape-dependent (helps moderate-N shapes with occupancy
// headroom, hurts huge-N shapes by cutting the row-group count) so the generator picks it per shape.
#ifndef NROW
#define NROW 1
#endif

// ============================================================================
// Main kernel: one subgroup (SG_SIZE lanes) owns NROW (n, bm) output elements.
//   global = [ceil(N/NROW) * SG_SIZE, BM, 1]   (BM = flattened batch*seq rows of the activation)
//   local  = [SG_SIZE, 1, 1]                   (one subgroup per work-group)
// K_SIZE and N_SIZE are static (the reduction and output-channel dims are fixed by the GGUF weight);
// only BM (activation rows) may be dynamic, and the dispatch sets global[1] == BM exactly, so the row
// index is taken straight from get_global_id(1) and needs no BM_SIZE bound (works for static & dynamic).
// n0 = (get_global_id(0)/SG_SIZE)*NROW is uniform across a subgroup, so the early-out and the
// sub_group_reduce_add are reached by all lanes together (no collective divergence).
// ============================================================================
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(fc_gguf_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,   // activations [BM, K]
    const __global uchar*       W,   // GGUF block weights [N, K] (opaque bytes)
          __global OUTPUT_TYPE* C    // output      [BM, N]
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const int n0   = (int)(get_global_id(0) / SG_SIZE) * NROW;   // first output row this subgroup owns
    const int lane = (int)get_sub_group_local_id();
    const int bm   = (int)get_global_id(1);

    if (n0 >= N_SIZE)
        return;

    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;

    // Stripe K-blocks across the subgroup lanes; for each block decode all NROW weight rows against the
    // SAME activation slice (L2-resident, reused across the rows). The (n0+r) guard keeps the unrolled
    // per-row reads in-bounds when N_SIZE is not a NROW multiple (loop-invariant per r -> hoisted).
    float partial[NROW];
    unroll_for (int r = 0; r < NROW; ++r)
        partial[r] = 0.0f;

    for (int kb = lane; kb < blocks_per_row; kb += SG_SIZE) {
        const uint blk_off = (uint)kb * GGUF_BLOCK_BYTES;
        const uint k0      = (uint)kb * GGUF_BLOCK_ELEM;
        unroll_for (int r = 0; r < NROW; ++r) {
            if ((n0 + r) >= N_SIZE)
                continue;
            const __global uchar* w_row = W + (uint)(n0 + r) * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
            partial[r] += FUNC_CALL(gguf_block_dot)(OPTIONAL_SHAPE_INFO_TENSOR
                                 w_row + blk_off, A, (uint)bm, k0);
        }
    }

    unroll_for (int r = 0; r < NROW; ++r) {
        const float total = sub_group_reduce_add(partial[r]);
        if (lane == 0 && (n0 + r) < N_SIZE) {
            // Output logical coords: BM splits into (b, f) exactly as the activation/output GET_INDEX,
            // and n is the channel (y) axis. The fused-ops idx_order in the generator must match this.
            const uint n     = (uint)(n0 + r);
            const uint out_b = (uint)bm / OUTPUT_FEATURE_NUM;
            const uint out_f = (uint)bm - out_b * OUTPUT_FEATURE_NUM;
#if HAS_FUSED_OPS
            OUTPUT_TYPE dequantized = TO_OUTPUT_TYPE(total);
            FUSED_OPS;
            C[FUNC_CALL(gguf_output_index)(OPTIONAL_SHAPE_INFO_TENSOR (uint)bm, n)] = FUSED_OPS_RESULT;
#else
            C[FUNC_CALL(gguf_output_index)(OPTIONAL_SHAPE_INFO_TENSOR (uint)bm, n)] = TO_OUTPUT_TYPE(total);
#endif
        }
    }
}
