// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// High-performance Q4_1 GEMV with Intel sub-group block reads (the "weight shuffle" layout).
//
// Computes  C[bm, n] = sum_k A[bm, k] * W[n, k]  for a Q4_1 weight that has been reordered at
// compile_model time into the SG-transposed ("shuffle") layout (see RepackGGUFWeightsShuffle).
//
// == Q4_1 format recap ==
//   GGUF block_q4_1 = { half d; half m; uint8_t qs[16]; }  (20 bytes, QK=32 weights)
//     qs[k] low  nibble = weight[k]      (k = 0..15)
//     qs[k] high nibble = weight[k+16]
//     dequant:  w = nibble * d + m       (nibble in 0..15, used DIRECTLY)
//
//   EIGHT consecutive Q4_1 blocks are grouped into one super-block of 256 weights. Each super-block
//   carries:
//     * 8 sub-blocks x 16 qs bytes            = 128 bytes  (weights,  "pqs")
//     * 8 fp16 scales d                       =  16 bytes  (scales,   "pd")
//     * 8 fp16 minima m                       =  16 bytes  (minima,   "pm")
//   which equals the original 8 x 20 = 160 bytes — a pure reorder.
//
//   The qs nibble layout is IDENTICAL to a Q4_0 / Q4_K sub-block, so the ACC_CHUNK dot-product macro
//   is reused verbatim. Because w = nibble * d + m we factor the per-sub-block scale/min out of the
//   inner loop:  acc += d_j * dot_j + m_j * isum_j  (min added with a "+" sign, per-block fp16).
//
// == Combined weight buffer (SAME total size as the native weight) ==
//   W = [ pqs | pd | pm ]
//   pqs:  total_blk * 128 bytes at offset 0            (weights, chunk-interleaved as Q4_K).
//   pd:   total_blk * 16 bytes at offset total_blk*128 (per-block scales d, SoA fp16).
//   pm:   total_blk * 16 bytes at offset total_blk*144 (per-block minima m, SoA fp16).
//     pd/pm entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes; field j occupies OPG fp16, lane lid's
//     value at j*OPG*2 + lid*2. One intel_sub_group_block_read_us8 fetches all 8 per lane.
//
// == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
//   Same technique as fc_gguf_q4k_sg.cl: KSPLIT sub-groups per work-group cover the SAME OPG output
//   rows but each owns a strided subset of the K-blocks; a single work-group barrier at the end
//   reduces the partials. KSPLIT=1 reproduces the original single-sub-group geometry exactly.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#pragma OPENCL EXTENSION cl_intel_subgroups        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short  : enable

#ifndef OPG
#define OPG 16  // = SG_SIZE = sub_group_size
#endif

#ifndef KSPLIT
#define KSPLIT 1  // sub-groups per work-group (K-split factor); set from CPP JIT
#endif

// Accumulate one 4-byte chunk (8 quantized weights: 4 low + 4 high nibbles) into the sub-block dot.
// Identical to fc_gguf_q4k_sg.cl.
#define ACC_CHUNK(dot, u, c, in)                                                 \
    do {                                                                         \
        uint _w = (u);                                                           \
        (dot) = fma((in)[(c)*4 + 0],      (float)( _w        & 0xFu), (dot));     \
        (dot) = fma((in)[(c)*4 + 1],      (float)((_w >>  8) & 0xFu), (dot));     \
        (dot) = fma((in)[(c)*4 + 2],      (float)((_w >> 16) & 0xFu), (dot));     \
        (dot) = fma((in)[(c)*4 + 3],      (float)((_w >> 24) & 0xFu), (dot));     \
        (dot) = fma((in)[16 + (c)*4 + 0], (float)((_w >>  4) & 0xFu), (dot));     \
        (dot) = fma((in)[16 + (c)*4 + 1], (float)((_w >> 12) & 0xFu), (dot));     \
        (dot) = fma((in)[16 + (c)*4 + 2], (float)((_w >> 20) & 0xFu), (dot));     \
        (dot) = fma((in)[16 + (c)*4 + 3], (float)((_w >> 28) & 0xFu), (dot));     \
    } while (0)

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(fc_gguf_q4_1_sg)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,       // activation [BM, K] (f16)
    const __global uchar*       W,       // shuffled weight buffer: [pqs | pd | pm]
          __global OUTPUT_TYPE* C        // output [BM, N]
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint nbpr = (uint)K_SIZE / 256u;             // K-blocks per row
    const uint h    = (uint)get_group_id(0);           // row-group
    const uint hh   = (uint)get_sub_group_local_id();  // lane = output row within group
    const uint bm   = (uint)get_global_id(1);          // activation row (BM)
    const uint sg   = (uint)get_local_id(2);           // sub-group / K-split index

    // Per-sub-group activation SLM (double-buffered) + KSPLIT reduction scratch.
    __local float slm_inp[KSPLIT * 2 * 256];
#if KSPLIT > 1
    __local float slm_red[KSPLIT * OPG];
#endif
    __local float* sbase = slm_inp + sg * (2 * 256);

    const uint total_blk = (uint)N_SIZE * nbpr;
    const uint off_pqs = 0u;
    const uint off_pd  = total_blk * 128u;
    const uint off_pm  = off_pd + total_blk * 16u;

    const __global uint*   pqs_u = (const __global uint*)(W + off_pqs);
    const __global ushort* pd_us = (const __global ushort*)(W + off_pd);
    const __global ushort* pm_us = (const __global ushort*)(W + off_pm);

    const __global INPUT0_TYPE* A_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;

    // Each sub-group processes a strided subset of the K-blocks.
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;   // current buffer

        // --- Issue all long-latency global loads first (weights + scales) ---
        uint pe_u = (h * nbpr + bid) * (OPG * 128 / 4);   // uints
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);  // sub 0,1
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);  // sub 2,3
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);  // sub 4,5
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);  // sub 6,7

        // 8 fp16 scales + 8 fp16 minima per lane (SoA), one wide read each.
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    // ushorts
        ushort8 dv = intel_sub_group_block_read_us8(pd_us + pd_off);
        ushort8 mv = intel_sub_group_block_read_us8(pm_us + pd_off);

        // --- Load 256 activations into SLM (each lane writes 16 elements at stride OPG) ---
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < 16u; ++i) {
            sb[i * OPG + hh] = (float)A_row[bid * 256u + i * OPG + hh];
        }

        // --- Decode 8 (scale, min) pairs: Q4_1 w = nibble * d + m => scale_j = d_j, minv_j = m_j.
        float scale[8], minv[8];
        {
            ushort dh[8], mh[8];
            dh[0]=dv.s0; dh[1]=dv.s1; dh[2]=dv.s2; dh[3]=dv.s3;
            dh[4]=dv.s4; dh[5]=dv.s5; dh[6]=dv.s6; dh[7]=dv.s7;
            mh[0]=mv.s0; mh[1]=mv.s1; mh[2]=mv.s2; mh[3]=mv.s3;
            mh[4]=mv.s4; mh[5]=mv.s5; mh[6]=mv.s6; mh[7]=mv.s7;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < 8; ++i) {
                scale[i] = (float)as_half(dh[i]);
                minv[i]  = (float)as_half(mh[i]);
            }
        }

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // --- Per-sub-block activation sums (weight independent) ---
        float isum[8];
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 8; ++j) {
            float s = 0.0f;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < 32; ++i)
                s += sb[j * 32 + i];
            isum[j] = s;
        }

        // --- Dot products + factored scale/min combine ---
        float dot;
        __local const float* in;

        in = sb + 0*32; dot = 0.0f;
        ACC_CHUNK(dot, w0.s0, 0, in); ACC_CHUNK(dot, w0.s1, 1, in);
        ACC_CHUNK(dot, w0.s2, 2, in); ACC_CHUNK(dot, w0.s3, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        ACC_CHUNK(dot, w0.s4, 0, in); ACC_CHUNK(dot, w0.s5, 1, in);
        ACC_CHUNK(dot, w0.s6, 2, in); ACC_CHUNK(dot, w0.s7, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        ACC_CHUNK(dot, w1.s0, 0, in); ACC_CHUNK(dot, w1.s1, 1, in);
        ACC_CHUNK(dot, w1.s2, 2, in); ACC_CHUNK(dot, w1.s3, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        ACC_CHUNK(dot, w1.s4, 0, in); ACC_CHUNK(dot, w1.s5, 1, in);
        ACC_CHUNK(dot, w1.s6, 2, in); ACC_CHUNK(dot, w1.s7, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        ACC_CHUNK(dot, w2.s0, 0, in); ACC_CHUNK(dot, w2.s1, 1, in);
        ACC_CHUNK(dot, w2.s2, 2, in); ACC_CHUNK(dot, w2.s3, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        ACC_CHUNK(dot, w2.s4, 0, in); ACC_CHUNK(dot, w2.s5, 1, in);
        ACC_CHUNK(dot, w2.s6, 2, in); ACC_CHUNK(dot, w2.s7, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        ACC_CHUNK(dot, w3.s0, 0, in); ACC_CHUNK(dot, w3.s1, 1, in);
        ACC_CHUNK(dot, w3.s2, 2, in); ACC_CHUNK(dot, w3.s3, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        ACC_CHUNK(dot, w3.s4, 0, in); ACC_CHUNK(dot, w3.s5, 1, in);
        ACC_CHUNK(dot, w3.s6, 2, in); ACC_CHUNK(dot, w3.s7, 3, in);
        acc = fma(scale[7], dot, acc); acc = fma(minv[7], isum[7], acc);
    }

    // --- Reduce KSPLIT partial sums; sub-group 0 writes the output. ---
#if KSPLIT > 1
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0)
        return;
    float sum = 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int g = 0; g < KSPLIT; ++g)
        sum += slm_red[g * OPG + hh];
    acc = sum;
#endif

    const uint n = h * OPG + hh;   // output row (= channel n)
    if (n < (uint)N_SIZE) {
        const uint out_b = bm / OUTPUT_FEATURE_NUM;
        const uint out_f = bm - out_b * OUTPUT_FEATURE_NUM;
#if HAS_FUSED_OPS
        OUTPUT_TYPE dequantized = TO_OUTPUT_TYPE(acc);
        FUSED_OPS;
        C[bm * (uint)N_SIZE + n] = FUSED_OPS_RESULT;
#else
        C[bm * (uint)N_SIZE + n] = TO_OUTPUT_TYPE(acc);
#endif
    }
}
