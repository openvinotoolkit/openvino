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
// One work-item owns one (n, bm) pair: it walks W's row `n` block-by-block, decodes each block into
// up to GGUF_BLOCK_ELEM half values, and multiplies them against the matching A elements. The block
// decoders mirror the canonical ggml reference (ggml-quants.c) and the CPU reference in the GGUF
// frontend (src/frontends/gguf/src/builders/dequantize.cpp).
//
// The packed GGUF source format is selected at JIT time by exactly one GGUF_IS_<TYPE> flag, together
// with GGUF_BLOCK_ELEM (logical elements per block) and GGUF_BLOCK_BYTES (bytes per block).
//
// Helper functions are wrapped in FUNC()/FUNC_CALL() so their names are decorated with the kernel
// entry point — multiple GGUF FC kernels (different shapes/formats) are batch-compiled into a single
// OpenCL program, and undecorated names would collide ("redefinition of ...").

#include "include/batch_headers/common.cl"

// Reconstruct a half from two little-endian bytes (GGUF is little-endian, as is every OV host/target).
inline half FUNC(gguf_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

// ============================================================================
// Per-format block decoders. Each fills `out[0 .. GGUF_BLOCK_ELEM)` with dequantised half values
// for the block starting at `blk`.
// ============================================================================

#if defined(GGUF_IS_Q4_0)
inline void FUNC(gguf_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        const int lo = (int)(qs[j] & 0x0F) - 8;
        const int hi = (int)(qs[j] >> 4) - 8;
        out[j]      = (half)(lo * (float)d);
        out[j + 16] = (half)(hi * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q8_0)
inline void FUNC(gguf_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(gguf_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    for (int j = 0; j < 32; ++j) {
        out[j] = (half)((float)qs[j] * (float)d);
    }
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
inline void FUNC(gguf_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;    // 12 bytes
    const __global uchar* qs     = blk + 16;   // 128 bytes
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
            out[o++] = (half)(d1 * (float)(qs[l] & 0x0F) - m1);
        }
        for (int l = 0; l < 32; ++l) {
            out[o++] = (half)(d2 * (float)(qs[l] >> 4) - m2);
        }
        qs += 32;
        is += 2;
    }
}
#endif

#if defined(GGUF_IS_Q5_K)
inline void FUNC(gguf_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(gguf_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(gguf_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;    // 12 bytes
    const __global uchar* qh     = blk + 16;   // 32 bytes (high bit-plane)
    const __global uchar* ql     = blk + 48;   // 128 bytes (low 4 bits)
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
            out[o++] = (half)(d1 * (float)q - m1);
        }
        for (int l = 0; l < 32; ++l) {
            const int q = (int)(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            out[o++] = (half)(d2 * (float)q - m2);
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}
#endif

#if defined(GGUF_IS_Q6_K)
inline void FUNC(gguf_decode_block)(const __global uchar* blk, __private half* out) {
    const __global uchar* ql = blk;            // 128 bytes (low 4 bits)
    const __global uchar* qh = blk + 128;      // 64 bytes (high 2 bits)
    const __global char*  sc = (const __global char*)(blk + 192);  // 16 signed scales
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk + 208);
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
        o  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}
#endif

// ============================================================================
// Main kernel: one work-item per (n, bm) output element.
//   global = [N_SIZE, BM, 1]   (BM = flattened batch*seq rows of the activation)
// K_SIZE and N_SIZE are static (the reduction and output-channel dims are fixed by the GGUF weight);
// only BM (activation rows) may be dynamic, and the dispatch sets global[1] == BM exactly, so the row
// index is taken straight from get_global_id(1) and needs no BM_SIZE bound (works for static & dynamic).
// ============================================================================
KERNEL(fc_gguf_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,   // activations [BM, K]
    const __global uchar*       W,   // GGUF block weights [N, K] (opaque bytes)
          __global OUTPUT_TYPE* C    // output      [BM, N]
)
{
    const int n  = (int)get_global_id(0);
    const int bm = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    const __global uchar* w_row = W + (uint)n * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
    const __global INPUT0_TYPE* a_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;
    half wvals[GGUF_BLOCK_ELEM];

    for (int kb = 0; kb < blocks_per_row; ++kb) {
        FUNC_CALL(gguf_decode_block)(w_row + (uint)kb * GGUF_BLOCK_BYTES, wvals);
        const int k0 = kb * GGUF_BLOCK_ELEM;
        for (int j = 0; j < GGUF_BLOCK_ELEM; ++j) {
            acc += (float)a_row[k0 + j] * (float)wvals[j];
        }
    }

    C[(uint)bm * (uint)N_SIZE + n] = TO_OUTPUT_TYPE(acc);
}
