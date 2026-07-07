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

// Reconstruct a half from two little-endian bytes (GGUF is little-endian, as is every OV host/target).
inline half FUNC(gguf_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

// ============================================================================
// Per-format streaming block dot. Each returns sum_{j in [0, GGUF_BLOCK_ELEM)} a[j] * dequant(blk[j])
// for the block starting at `blk` against the activation slice `a`, accumulating in float without
// materialising the dequantised block (keeps register pressure low so SG_SIZE lanes stay resident).
// ============================================================================

#if defined(GGUF_IS_Q4_0)
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    float acc = 0.0f;
    for (int j = 0; j < 16; ++j) {
        const int lo = (int)(qs[j] & 0x0F) - 8;
        const int hi = (int)(qs[j] >> 4) - 8;
        acc += (float)a[j]      * ((float)lo * d);
        acc += (float)a[j + 16] * ((float)hi * d);
    }
    return acc;
}
#endif

#if defined(GGUF_IS_Q8_0)
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
    const float d = (float)FUNC_CALL(gguf_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    float acc = 0.0f;
    for (int j = 0; j < 32; ++j) {
        acc += (float)a[j] * ((float)qs[j] * d);
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
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
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
            const float av = (float)a[ai + l];
            sq1 += av * (float)(qs[l] & 0x0F);
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = (float)a[ai + 32 + l];
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
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
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
            const float av = (float)a[ai + l];
            const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
            sq1 += av * (float)q;
            sa1 += av;
        }
        for (int l = 0; l < 32; ++l) {
            const float av = (float)a[ai + 32 + l];
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
inline float FUNC(gguf_block_dot)(const __global uchar* blk, const __global INPUT0_TYPE* a) {
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
            acc1 += (float)a[o + l + 0]  * (d * (float)sc[is + 0] * q1);
            acc2 += (float)a[o + l + 32] * (d * (float)sc[is + 2] * q2);
            acc3 += (float)a[o + l + 64] * (d * (float)sc[is + 4] * q3);
            acc4 += (float)a[o + l + 96] * (d * (float)sc[is + 6] * q4);
        }
        o  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc1 + acc2 + acc3 + acc4;
}
#endif

// ============================================================================
// Main kernel: one subgroup (SG_SIZE lanes) per (n, bm) output element.
//   global = [N_SIZE * SG_SIZE, BM, 1]   (BM = flattened batch*seq rows of the activation)
//   local  = [SG_SIZE, 1, 1]             (one subgroup per work-group)
// K_SIZE and N_SIZE are static (the reduction and output-channel dims are fixed by the GGUF weight);
// only BM (activation rows) may be dynamic, and the dispatch sets global[1] == BM exactly, so the row
// index is taken straight from get_global_id(1) and needs no BM_SIZE bound (works for static & dynamic).
// n = get_global_id(0)/SG_SIZE is uniform across a subgroup, so the early-out and the
// sub_group_reduce_add are reached by all lanes together (no collective divergence).
// ============================================================================
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
KERNEL(fc_gguf_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,   // activations [BM, K]
    const __global uchar*       W,   // GGUF block weights [N, K] (opaque bytes)
          __global OUTPUT_TYPE* C    // output      [BM, N]
)
{
    const int n    = (int)(get_global_id(0) / SG_SIZE);
    const int lane = (int)get_sub_group_local_id();
    const int bm   = (int)get_global_id(1);

    if (n >= N_SIZE)
        return;

    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    const __global uchar* w_row = W + (uint)n * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
    const __global INPUT0_TYPE* a_row = A + (uint)bm * (uint)K_SIZE;

    // Stripe row `n`'s blocks across the subgroup lanes; each lane streams its blocks' dot product.
    float partial = 0.0f;
    for (int kb = lane; kb < blocks_per_row; kb += SG_SIZE) {
        partial += FUNC_CALL(gguf_block_dot)(w_row + (uint)kb * GGUF_BLOCK_BYTES,
                                             a_row + (uint)kb * GGUF_BLOCK_ELEM);
    }

    const float total = sub_group_reduce_add(partial);
    if (lane == 0)
        C[(uint)bm * (uint)N_SIZE + n] = TO_OUTPUT_TYPE(total);
}
