// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF weight-shuffle GEMV (decode / small M) for Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and Q8_0.
//
// One source for all six formats: the host defines exactly one GG_FMT_* macro per kernel and gives each
// format its own entry point (fc_gguf_<fmt>_sg). Every format reads the SG-transposed, plane-separated weight
// produced by RepackGGUFWeightsShuffle with intel_sub_group_block_read and keeps one complete output row per
// lane; the per-format sections below document the plane layouts.

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#pragma OPENCL EXTENSION cl_intel_subgroups        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short  : enable
#if GG_FMT_Q6_K
#pragma OPENCL EXTENSION cl_intel_subgroups_char   : enable
#endif

#ifndef OPG
#define OPG 16  // = SG_SIZE = sub_group_size
#endif

#ifndef KSPLIT
#define KSPLIT 1  // sub-groups per work-group (K-split factor); set from CPP JIT
#endif

#if GG_FMT_Q4_0 || GG_FMT_Q4_1 || GG_FMT_Q4_K
// Accumulate one 4-byte chunk (8 quantized weights: 4 low + 4 high nibbles) into the sub-block dot.
// `in` points at slm base of the sub-block, `c` is the chunk index (0..3). Low nibbles map to
// in[c*4 + b], high nibbles to in[16 + c*4 + b]. Shared by the Q4_0 / Q4_1 / Q4_K kernels.
#define Q4_ACC_CHUNK(dot, u, c, in)                                                 \
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
#endif

#if GG_FMT_Q4_0
// ==================================================================================================
// High-performance Q4_0 GEMV with Intel sub-group block reads (the "weight shuffle" layout).
//
// Computes  C[bm, n] = sum_k A[bm, k] * W[n, k]  for a Q4_0 weight that has been reordered at
// compile_model time into the SG-transposed ("shuffle") layout (see RepackGGUFWeightsShuffle).
//
// == Q4_0 format recap ==
//   GGUF block_q4_0 = { half d; uint8_t qs[16]; }  (18 bytes, QK=32 weights)
//     qs[k] low  nibble = weight[k]      (k = 0..15)
//     qs[k] high nibble = weight[k+16]
//     dequant:  w = (nibble - 8) * d
//
//   To reuse the 256-weight ("super-block") machinery of the Q4_K section EIGHT consecutive Q4_0
//   blocks are grouped into one super-block of 256 weights. Each super-block therefore carries:
//     * 8 sub-blocks x 16 qs bytes            = 128 bytes  (weights,  "pqs")
//     * 8 fp16 scales d                       =  16 bytes  (scales,   "pd")
//   which equals the original 8 x 18 = 144 bytes — a pure reorder.
//
//   The qs nibble layout is IDENTICAL to a Q4_K sub-block, so the Q4_ACC_CHUNK dot-product macro is
//   reused verbatim. Because w = (nibble - 8) * d we factor the per-sub-block scale/bias out of the
//   inner loop:  acc += d_j * dot_j - (8 * d_j) * isum_j.
//
// == Combined weight buffer (SAME total size as the native weight) ==
//   W = [ pqs | pd ]
//   pqs:  total_blk * 128 bytes at offset 0.
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 128 bytes (2048 B)
//     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
//   pd:   total_blk * 16 bytes at offset total_blk*128.
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B), SoA fields:
//       field j (0..7) occupies OPG fp16, lane lid's d at j*OPG*2 + lid*2.
//     One intel_sub_group_block_read_us8 fetches all 8 scales per lane.
//
// == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
//   Same technique as the Q4_K section: KSPLIT sub-groups per work-group cover the SAME OPG output
//   rows but each owns a strided subset of the K-blocks; a single work-group barrier at the end
//   reduces the partials. Total live threads = (N/OPG) * KSPLIT. KSPLIT=1 reproduces the original
//   single-sub-group geometry exactly.

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(fc_gguf_q4_0_sg)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,       // activation [BM, K] (f16)
    const __global uchar*       W,       // shuffled weight buffer: [pqs | pd]
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

    const __global uint*   pqs_u = (const __global uint*)(W + off_pqs);
    const __global ushort* pd_us = (const __global ushort*)(W + off_pd);

    const __global INPUT0_TYPE* A_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;

    // Each sub-group processes a strided subset of the K-blocks.
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;   // current buffer

        // --- Issue all long-latency global loads first (weights + scales) ---
        uint pe_u = (h * nbpr + bid) * (OPG * 128 / 4);   // uints
        uint8 w0 = BLOCK_READN(uint, 8, pqs_u, pe_u +   0);  // sub 0,1
        uint8 w1 = BLOCK_READN(uint, 8, pqs_u, pe_u + 128);  // sub 2,3
        uint8 w2 = BLOCK_READN(uint, 8, pqs_u, pe_u + 256);  // sub 4,5
        uint8 w3 = BLOCK_READN(uint, 8, pqs_u, pe_u + 384);  // sub 6,7

        // 8 fp16 scales per lane (SoA), fetched with ONE wide block read.
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    // ushorts
        ushort8 dv = BLOCK_READN(ushort, 8, pd_us, pd_off);

        // --- Load 256 activations into SLM (each lane writes 16 elements at stride OPG) ---
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < 16u; ++i) {
            sb[i * OPG + hh] = (float)A_row[bid * 256u + i * OPG + hh];
        }

        // --- Decode 8 (scale, bias) pairs: Q4_0 w = (nibble - 8) * d => scale_j = d_j, bias_j = 8*d_j.
        float scale[8], minv[8];
        {
            ushort dh[8];
            dh[0]=dv.s0; dh[1]=dv.s1; dh[2]=dv.s2; dh[3]=dv.s3;
            dh[4]=dv.s4; dh[5]=dv.s5; dh[6]=dv.s6; dh[7]=dv.s7;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < 8; ++i) {
                float d  = (float)as_half(dh[i]);
                scale[i] = d;
                minv[i]  = 8.0f * d;
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

        // --- Dot products + factored scale/bias combine ---
        float dot;
        __local const float* in;

        in = sb + 0*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w0.s0, 0, in); Q4_ACC_CHUNK(dot, w0.s1, 1, in);
        Q4_ACC_CHUNK(dot, w0.s2, 2, in); Q4_ACC_CHUNK(dot, w0.s3, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(-minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w0.s4, 0, in); Q4_ACC_CHUNK(dot, w0.s5, 1, in);
        Q4_ACC_CHUNK(dot, w0.s6, 2, in); Q4_ACC_CHUNK(dot, w0.s7, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(-minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w1.s0, 0, in); Q4_ACC_CHUNK(dot, w1.s1, 1, in);
        Q4_ACC_CHUNK(dot, w1.s2, 2, in); Q4_ACC_CHUNK(dot, w1.s3, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(-minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w1.s4, 0, in); Q4_ACC_CHUNK(dot, w1.s5, 1, in);
        Q4_ACC_CHUNK(dot, w1.s6, 2, in); Q4_ACC_CHUNK(dot, w1.s7, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(-minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w2.s0, 0, in); Q4_ACC_CHUNK(dot, w2.s1, 1, in);
        Q4_ACC_CHUNK(dot, w2.s2, 2, in); Q4_ACC_CHUNK(dot, w2.s3, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(-minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w2.s4, 0, in); Q4_ACC_CHUNK(dot, w2.s5, 1, in);
        Q4_ACC_CHUNK(dot, w2.s6, 2, in); Q4_ACC_CHUNK(dot, w2.s7, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(-minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w3.s0, 0, in); Q4_ACC_CHUNK(dot, w3.s1, 1, in);
        Q4_ACC_CHUNK(dot, w3.s2, 2, in); Q4_ACC_CHUNK(dot, w3.s3, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(-minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w3.s4, 0, in); Q4_ACC_CHUNK(dot, w3.s5, 1, in);
        Q4_ACC_CHUNK(dot, w3.s6, 2, in); Q4_ACC_CHUNK(dot, w3.s7, 3, in);
        acc = fma(scale[7], dot, acc); acc = fma(-minv[7], isum[7], acc);
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

#elif GG_FMT_Q4_1
// ==================================================================================================
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
//   The qs nibble layout is IDENTICAL to a Q4_0 / Q4_K sub-block, so the Q4_ACC_CHUNK dot-product macro
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
//   Same technique as the Q4_K section: KSPLIT sub-groups per work-group cover the SAME OPG output
//   rows but each owns a strided subset of the K-blocks; a single work-group barrier at the end
//   reduces the partials. KSPLIT=1 reproduces the original single-sub-group geometry exactly.

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

    // Per-sub-group activation SLM (single-buffered -- see the end-of-iteration barrier below) +
    // KSPLIT reduction scratch. Halves the SLM footprint vs the earlier double-buffered layout, which
    // was the proven Peak-XVE-Threads-Occupancy limiter (SLM Use Limit == measured occupancy at
    // 77.6-80%): fewer bytes/work-group lets more work-groups stay co-resident on one XVE.
    __local float slm_inp[KSPLIT * 256];
#if KSPLIT > 1
    __local float slm_red[KSPLIT * OPG];
#endif
    __local float* sbase = slm_inp + sg * 256;

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
        __local float* sb = sbase;

        // --- Issue all long-latency global loads first (weights + scales) ---
        uint pe_u = (h * nbpr + bid) * (OPG * 128 / 4);   // uints
        uint8 w0 = BLOCK_READN(uint, 8, pqs_u, pe_u +   0);  // sub 0,1
        uint8 w1 = BLOCK_READN(uint, 8, pqs_u, pe_u + 128);  // sub 2,3
        uint8 w2 = BLOCK_READN(uint, 8, pqs_u, pe_u + 256);  // sub 4,5
        uint8 w3 = BLOCK_READN(uint, 8, pqs_u, pe_u + 384);  // sub 6,7

        // 8 fp16 scales + 8 fp16 minima per lane (SoA), one wide read each.
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    // ushorts
        ushort8 dv = BLOCK_READN(ushort, 8, pd_us, pd_off);
        ushort8 mv = BLOCK_READN(ushort, 8, pm_us, pd_off);

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
        Q4_ACC_CHUNK(dot, w0.s0, 0, in); Q4_ACC_CHUNK(dot, w0.s1, 1, in);
        Q4_ACC_CHUNK(dot, w0.s2, 2, in); Q4_ACC_CHUNK(dot, w0.s3, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w0.s4, 0, in); Q4_ACC_CHUNK(dot, w0.s5, 1, in);
        Q4_ACC_CHUNK(dot, w0.s6, 2, in); Q4_ACC_CHUNK(dot, w0.s7, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w1.s0, 0, in); Q4_ACC_CHUNK(dot, w1.s1, 1, in);
        Q4_ACC_CHUNK(dot, w1.s2, 2, in); Q4_ACC_CHUNK(dot, w1.s3, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w1.s4, 0, in); Q4_ACC_CHUNK(dot, w1.s5, 1, in);
        Q4_ACC_CHUNK(dot, w1.s6, 2, in); Q4_ACC_CHUNK(dot, w1.s7, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w2.s0, 0, in); Q4_ACC_CHUNK(dot, w2.s1, 1, in);
        Q4_ACC_CHUNK(dot, w2.s2, 2, in); Q4_ACC_CHUNK(dot, w2.s3, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w2.s4, 0, in); Q4_ACC_CHUNK(dot, w2.s5, 1, in);
        Q4_ACC_CHUNK(dot, w2.s6, 2, in); Q4_ACC_CHUNK(dot, w2.s7, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w3.s0, 0, in); Q4_ACC_CHUNK(dot, w3.s1, 1, in);
        Q4_ACC_CHUNK(dot, w3.s2, 2, in); Q4_ACC_CHUNK(dot, w3.s3, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w3.s4, 0, in); Q4_ACC_CHUNK(dot, w3.s5, 1, in);
        Q4_ACC_CHUNK(dot, w3.s6, 2, in); Q4_ACC_CHUNK(dot, w3.s7, 3, in);
        acc = fma(scale[7], dot, acc); acc = fma(minv[7], isum[7], acc);

        // sb is single-buffered now: the NEXT iteration's activation writes (top of loop) must not
        // race this iteration's reads above (isum / ACC_CHUNK) -- the double-buffered layout used to
        // make this safe by construction (disjoint banks); this barrier replaces that guarantee.
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
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

#elif GG_FMT_Q4_K
// ==================================================================================================
// High-performance Q4_K GEMV with Intel sub-group block reads (the "weight shuffle" layout).
//
// Computes  C[bm, n] = sum_k A[bm, k] * W[n, k]  for a Q4_K weight that has been reordered at
// compile_model time into the SG-transposed ("shuffle") layout (see RepackGGUFWeightsShuffle):
//
//   Group N-rows by OPG = SG_SIZE = 16. For row-group h and K-block bid the OPG blocks are
//   interleaved in 4-byte chunks so a single intel_sub_group_block_read delivers 4 contiguous
//   weight bytes to each of the 16 lanes -> fully coalesced weight loads.
//
//   pqs_T:  total_blk * 128 bytes (== raw pqs size, only layout differs)
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 128 bytes (2048 B)
//     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
//     Lane lid gets pqs[blk*128 + j*16 + c*4 .. +3], blk=(h*OPG+lid)*nbpr+bid.
//   psl_T:  total_blk * 16 bytes
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B), SoA fields:
//       [ sl_u32*OPG=64B | ml_u32*OPG=64B | sh_u16*OPG=32B | mh_u16*OPG=32B |
//         d_u16*OPG=32B  | dmin_u16*OPG=32B ]
//
// Each lane owns ONE complete output row, so the running dot lives in a private register `acc`.
//
// == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
//   The baseline dispatch used ONE sub-group per work-group, so the number of concurrent hardware
//   threads == N/OPG. For small N this leaves the machine badly under-occupied, the dominant
//   bottleneck for this memory-bound kernel. We put KSPLIT sub-groups in each work-group (local =
//   OPG x 1 x KSPLIT). All KSPLIT sub-groups of a group cover the SAME OPG output rows but each owns
//   a strided subset of the K-blocks (sub-group sg handles bid = sg, sg+KSPLIT, ...). Each keeps a
//   private partial `acc`; at the end the KSPLIT partials are reduced through a small SLM buffer with
//   a single work-group barrier and sub-group 0 writes the output. Total live threads =
//   (N/OPG) * KSPLIT. KSPLIT=1 reproduces the original single-sub-group geometry exactly.
//
// Weights are read with wide intel_sub_group_block_read8 (512 B/call) to maximise in-flight loads for
// this memory-bound kernel, all long-latency global loads issued BEFORE the SLM sync so they overlap
// the barrier and the scalar decode work. The per-sub-block quantized scale/min are factored OUT of
// the inner loop: for sub-block j we compute  dot_j = sum a_k * qnibble_k  and  isum_j = sum a_k
// (weight-independent), then combine once as  acc += scale_j*dot_j - min_j*isum_j. The per-sub-group
// activation SLM is DOUBLE-BUFFERED so exactly one sub_group_barrier per K-block suffices.

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(fc_gguf_q4k_sg)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,       // activation [BM, K] (f16)
    const __global uchar*       W,       // shuffled weight buffer: [pqs | psl]
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
    const uint off_psl = total_blk * 128u;

    const __global uchar* PQS = W + off_pqs;
    const __global uchar* PSL = W + off_psl;

    const __global uint*   pqs_u  = (const __global uint*)PQS;
    const __global uint*   psl_pu = (const __global uint*)PSL;
    const __global ushort* psl_us = (const __global ushort*)PSL;

    const __global INPUT0_TYPE* A_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;

    // Each sub-group processes a strided subset of the K-blocks.
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;   // current buffer

        // --- Issue all long-latency global loads first (weights + scales) ---
        uint pe_u = (h * nbpr + bid) * (OPG * 128 / 4);   // uints
        uint8 w0 = BLOCK_READN(uint, 8, pqs_u, pe_u +   0);  // sub 0,1
        uint8 w1 = BLOCK_READN(uint, 8, pqs_u, pe_u + 128);  // sub 2,3
        uint8 w2 = BLOCK_READN(uint, 8, pqs_u, pe_u + 256);  // sub 4,5
        uint8 w3 = BLOCK_READN(uint, 8, pqs_u, pe_u + 384);  // sub 6,7

        uint   psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);   // uints
        uint   psl_us_off = (h * nbpr + bid) * (OPG * 16 / 2);   // ushort
        uint   sl     = intel_sub_group_block_read   (psl_pu + psl_pu_off +  0);
        uint   ml     = intel_sub_group_block_read   (psl_pu + psl_pu_off + 16);
        ushort sh     = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 64);
        ushort mh     = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 80);
        ushort d_raw  = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 96);
        ushort dn_raw = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 112);

        // --- Load 256 activations into SLM (each lane writes 16 elements at stride OPG) ---
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < 16u; ++i) {
            sb[i * OPG + hh] = (float)A_row[bid * 256u + i * OPG + hh];
        }

        // --- Decode 8 (scale, min) pairs (overlaps outstanding loads) ---
        float scale[8], minv[8];
        {
            float d    = (float)as_half(d_raw);
            float dmin = (float)as_half(dn_raw);
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < 8; ++i) {
                uint sq = ((sl >> (i * 4)) & 0xFu) | ((uint)((sh >> (i * 2)) & 0x3u) << 4);
                uint mq = ((ml >> (i * 4)) & 0xFu) | ((uint)((mh >> (i * 2)) & 0x3u) << 4);
                scale[i] = (float)sq * d;
                minv[i]  = (float)mq * dmin;
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
        Q4_ACC_CHUNK(dot, w0.s0, 0, in); Q4_ACC_CHUNK(dot, w0.s1, 1, in);
        Q4_ACC_CHUNK(dot, w0.s2, 2, in); Q4_ACC_CHUNK(dot, w0.s3, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(-minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w0.s4, 0, in); Q4_ACC_CHUNK(dot, w0.s5, 1, in);
        Q4_ACC_CHUNK(dot, w0.s6, 2, in); Q4_ACC_CHUNK(dot, w0.s7, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(-minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w1.s0, 0, in); Q4_ACC_CHUNK(dot, w1.s1, 1, in);
        Q4_ACC_CHUNK(dot, w1.s2, 2, in); Q4_ACC_CHUNK(dot, w1.s3, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(-minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w1.s4, 0, in); Q4_ACC_CHUNK(dot, w1.s5, 1, in);
        Q4_ACC_CHUNK(dot, w1.s6, 2, in); Q4_ACC_CHUNK(dot, w1.s7, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(-minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w2.s0, 0, in); Q4_ACC_CHUNK(dot, w2.s1, 1, in);
        Q4_ACC_CHUNK(dot, w2.s2, 2, in); Q4_ACC_CHUNK(dot, w2.s3, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(-minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w2.s4, 0, in); Q4_ACC_CHUNK(dot, w2.s5, 1, in);
        Q4_ACC_CHUNK(dot, w2.s6, 2, in); Q4_ACC_CHUNK(dot, w2.s7, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(-minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w3.s0, 0, in); Q4_ACC_CHUNK(dot, w3.s1, 1, in);
        Q4_ACC_CHUNK(dot, w3.s2, 2, in); Q4_ACC_CHUNK(dot, w3.s3, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(-minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        Q4_ACC_CHUNK(dot, w3.s4, 0, in); Q4_ACC_CHUNK(dot, w3.s5, 1, in);
        Q4_ACC_CHUNK(dot, w3.s6, 2, in); Q4_ACC_CHUNK(dot, w3.s7, 3, in);
        acc = fma(scale[7], dot, acc); acc = fma(-minv[7], isum[7], acc);
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

#elif GG_FMT_Q5_K
// ==================================================================================================
// High-performance Q5_K GEMV with Intel sub-group block reads (the "weight shuffle" layout).
//
// Computes  C[bm, n] = sum_k A[bm, k] * W[n, k]  for a Q5_K weight that has been reordered at
// compile_model time into the SG-transposed ("shuffle") layout (see RepackGGUFWeightsShuffle).
// This is the Q5_K counterpart of the Q4_K / Q6_K sections: Q5_K == Q4_K plus one
// extra high bit per weight (range 0..31), stored in a separate plane.
//
//   Group N-rows by OPG = SG_SIZE = 16. For row-group h and K-block bid the OPG blocks are
//   interleaved in 4-byte chunks so a single intel_sub_group_block_read delivers 4 contiguous
//   weight bytes to each of the 16 lanes -> fully coalesced weight loads.
//
//   pqs_T:  total_blk * 128 bytes (packed 4-bit low quants, SAME packing/layout as Q4_K)
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 128 bytes (2048 B)
//     Sub-block j at j*OPG*16, chunk c (0..3) at j*OPG*16 + c*OPG*4.
//   pqh_T:  total_blk * 32 bytes (packed high bit, 1 bit/weight, 4 bytes/sub-block)
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 32 bytes (512 B)
//     Sub-block j's 4-byte qh word for all OPG lanes at j*OPG*4 (lane lid's word at j*OPG*4 + lid*4)
//     -> one intel_sub_group_block_read8 fetches all 8 sub-block qh words at once.
//   psl_T:  total_blk * 16 bytes (identical SoA layout to Q4_K)
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B), SoA fields:
//       [ sl_u32*OPG=64B | ml_u32*OPG=64B | sh_u16*OPG=32B | mh_u16*OPG=32B |
//         d_u16*OPG=32B  | dmin_u16*OPG=32B ]
//
// Each lane owns ONE complete output row, so the running dot lives in a private register `acc`.
//
// == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
//   The baseline dispatch used ONE sub-group per work-group, so the number of concurrent hardware
//   threads == N/OPG. For small N this leaves the machine badly under-occupied, the dominant
//   bottleneck for this memory-bound kernel. We put KSPLIT sub-groups in each work-group (local =
//   OPG x 1 x KSPLIT). All KSPLIT sub-groups of a group cover the SAME OPG output rows but each owns
//   a strided subset of the K-blocks (sub-group sg handles bid = sg, sg+KSPLIT, ...). Each keeps a
//   private partial `acc`; at the end the KSPLIT partials are reduced through a small SLM buffer with
//   a single work-group barrier and sub-group 0 writes the output. Total live threads =
//   (N/OPG) * KSPLIT. KSPLIT=1 reproduces the original single-sub-group geometry exactly.
//
// Weights are read with wide intel_sub_group_block_read8 (512 B/call) to maximise in-flight loads for
// this memory-bound kernel, all long-latency global loads issued BEFORE the SLM sync so they overlap
// the barrier and the scalar decode work. The per-sub-block quantized scale/min are factored OUT of
// the inner loop: for sub-block j we compute  dot_j = sum a_k * weight_k  and  isum_j = sum a_k
// (weight-independent), then combine once as  acc += scale_j*dot_j - min_j*isum_j. The per-sub-group
// activation SLM is DOUBLE-BUFFERED so exactly one sub_group_barrier per K-block suffices.

// Accumulate one 4-byte qs chunk (8 quantized 5-bit weights: 4 low + 4 high nibbles, i.e. weight
// positions k=c*4..c*4+3 for the low half and k+16=c*4+16..c*4+19 for the high half) plus the
// corresponding high bits extracted from the 32-bit qh word for this sub-block, into the sub-block dot.
//
// Derivation of the high-bit shift (see RepackGGUFWeightsShuffle / gemv_q5k_shuffle):
//   low-half weight index  wi = c*4+i    -> byte_idx = i, bit_idx = c
//   high-half weight index wi = c*4+i+16 -> byte_idx = i, bit_idx = c+4
//   high_bit = (qh_word >> (byte_idx*8 + bit_idx)) & 1
#define Q5K_ACC_CHUNK(dot, u, qhw, c, in)                                        \
    do {                                                                         \
        uint _w = (u);                                                           \
        (dot) = fma((in)[(c)*4 + 0],                                             \
                    (float)(( _w        & 0xFu) |                                \
                            ((((qhw) >> (0*8 + (c)    )) & 1u) << 4)), (dot));    \
        (dot) = fma((in)[(c)*4 + 1],                                             \
                    (float)(((_w >>  8) & 0xFu) |                                \
                            ((((qhw) >> (1*8 + (c)    )) & 1u) << 4)), (dot));    \
        (dot) = fma((in)[(c)*4 + 2],                                             \
                    (float)(((_w >> 16) & 0xFu) |                                \
                            ((((qhw) >> (2*8 + (c)    )) & 1u) << 4)), (dot));    \
        (dot) = fma((in)[(c)*4 + 3],                                             \
                    (float)(((_w >> 24) & 0xFu) |                                \
                            ((((qhw) >> (3*8 + (c)    )) & 1u) << 4)), (dot));    \
        (dot) = fma((in)[16 + (c)*4 + 0],                                        \
                    (float)(((_w >>  4) & 0xFu) |                                \
                            ((((qhw) >> (0*8 + (c) + 4)) & 1u) << 4)), (dot));    \
        (dot) = fma((in)[16 + (c)*4 + 1],                                        \
                    (float)(((_w >> 12) & 0xFu) |                                \
                            ((((qhw) >> (1*8 + (c) + 4)) & 1u) << 4)), (dot));    \
        (dot) = fma((in)[16 + (c)*4 + 2],                                        \
                    (float)(((_w >> 20) & 0xFu) |                                \
                            ((((qhw) >> (2*8 + (c) + 4)) & 1u) << 4)), (dot));    \
        (dot) = fma((in)[16 + (c)*4 + 3],                                        \
                    (float)(((_w >> 28) & 0xFu) |                                \
                            ((((qhw) >> (3*8 + (c) + 4)) & 1u) << 4)), (dot));    \
    } while (0)

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(fc_gguf_q5k_sg)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,       // activation [BM, K] (f16)
    const __global uchar*       W,       // shuffled weight buffer: [pqs | pqh | psl]
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
    const uint off_pqh = total_blk * 128u;
    const uint off_psl = off_pqh + total_blk * 32u;

    const __global uchar* PQS = W + off_pqs;
    const __global uchar* PQH = W + off_pqh;
    const __global uchar* PSL = W + off_psl;

    const __global uint*   pqs_u  = (const __global uint*)PQS;
    const __global uint*   pqh_u  = (const __global uint*)PQH;
    const __global uint*   psl_pu = (const __global uint*)PSL;
    const __global ushort* psl_us = (const __global ushort*)PSL;

    const __global INPUT0_TYPE* A_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;

    // Each sub-group processes a strided subset of the K-blocks.
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;   // current buffer

        // --- Issue all long-latency global loads first (weights + high bits + scales) ---
        uint pqs_e = (h * nbpr + bid) * (OPG * 128 / 4);   // uints
        uint8 w0 = BLOCK_READN(uint, 8, pqs_u, pqs_e +   0);  // sub 0,1
        uint8 w1 = BLOCK_READN(uint, 8, pqs_u, pqs_e + 128);  // sub 2,3
        uint8 w2 = BLOCK_READN(uint, 8, pqs_u, pqs_e + 256);  // sub 4,5
        uint8 w3 = BLOCK_READN(uint, 8, pqs_u, pqs_e + 384);  // sub 6,7

        uint pqh_e = (h * nbpr + bid) * (OPG * 32 / 4);    // uints
        uint8 hq = BLOCK_READN(uint, 8, pqh_u, pqh_e);  // 8 sub-block qh words

        uint   psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);   // uints
        uint   psl_us_off = (h * nbpr + bid) * (OPG * 16 / 2);   // ushort
        uint   sl     = intel_sub_group_block_read   (psl_pu + psl_pu_off +  0);
        uint   ml     = intel_sub_group_block_read   (psl_pu + psl_pu_off + 16);
        ushort sh     = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 64);
        ushort mh     = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 80);
        ushort d_raw  = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 96);
        ushort dn_raw = BLOCK_READN(ushort, 1, psl_us, psl_us_off + 112);

        // --- Load 256 activations into SLM (each lane writes 16 elements at stride OPG) ---
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < 16u; ++i) {
            sb[i * OPG + hh] = (float)A_row[bid * 256u + i * OPG + hh];
        }

        // --- Decode 8 (scale, min) pairs (overlaps outstanding loads) ---
        float scale[8], minv[8];
        {
            float d    = (float)as_half(d_raw);
            float dmin = (float)as_half(dn_raw);
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < 8; ++i) {
                uint sq = ((sl >> (i * 4)) & 0xFu) | ((uint)((sh >> (i * 2)) & 0x3u) << 4);
                uint mq = ((ml >> (i * 4)) & 0xFu) | ((uint)((mh >> (i * 2)) & 0x3u) << 4);
                scale[i] = (float)sq * d;
                minv[i]  = (float)mq * dmin;
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
        Q5K_ACC_CHUNK(dot, w0.s0, hq.s0, 0, in); Q5K_ACC_CHUNK(dot, w0.s1, hq.s0, 1, in);
        Q5K_ACC_CHUNK(dot, w0.s2, hq.s0, 2, in); Q5K_ACC_CHUNK(dot, w0.s3, hq.s0, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(-minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        Q5K_ACC_CHUNK(dot, w0.s4, hq.s1, 0, in); Q5K_ACC_CHUNK(dot, w0.s5, hq.s1, 1, in);
        Q5K_ACC_CHUNK(dot, w0.s6, hq.s1, 2, in); Q5K_ACC_CHUNK(dot, w0.s7, hq.s1, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(-minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        Q5K_ACC_CHUNK(dot, w1.s0, hq.s2, 0, in); Q5K_ACC_CHUNK(dot, w1.s1, hq.s2, 1, in);
        Q5K_ACC_CHUNK(dot, w1.s2, hq.s2, 2, in); Q5K_ACC_CHUNK(dot, w1.s3, hq.s2, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(-minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        Q5K_ACC_CHUNK(dot, w1.s4, hq.s3, 0, in); Q5K_ACC_CHUNK(dot, w1.s5, hq.s3, 1, in);
        Q5K_ACC_CHUNK(dot, w1.s6, hq.s3, 2, in); Q5K_ACC_CHUNK(dot, w1.s7, hq.s3, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(-minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        Q5K_ACC_CHUNK(dot, w2.s0, hq.s4, 0, in); Q5K_ACC_CHUNK(dot, w2.s1, hq.s4, 1, in);
        Q5K_ACC_CHUNK(dot, w2.s2, hq.s4, 2, in); Q5K_ACC_CHUNK(dot, w2.s3, hq.s4, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(-minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        Q5K_ACC_CHUNK(dot, w2.s4, hq.s5, 0, in); Q5K_ACC_CHUNK(dot, w2.s5, hq.s5, 1, in);
        Q5K_ACC_CHUNK(dot, w2.s6, hq.s5, 2, in); Q5K_ACC_CHUNK(dot, w2.s7, hq.s5, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(-minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        Q5K_ACC_CHUNK(dot, w3.s0, hq.s6, 0, in); Q5K_ACC_CHUNK(dot, w3.s1, hq.s6, 1, in);
        Q5K_ACC_CHUNK(dot, w3.s2, hq.s6, 2, in); Q5K_ACC_CHUNK(dot, w3.s3, hq.s6, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(-minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        Q5K_ACC_CHUNK(dot, w3.s4, hq.s7, 0, in); Q5K_ACC_CHUNK(dot, w3.s5, hq.s7, 1, in);
        Q5K_ACC_CHUNK(dot, w3.s6, hq.s7, 2, in); Q5K_ACC_CHUNK(dot, w3.s7, hq.s7, 3, in);
        acc = fma(scale[7], dot, acc); acc = fma(-minv[7], isum[7], acc);
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

#elif GG_FMT_Q6_K
// ==================================================================================================
// High-performance Q6_K GEMV with Intel sub-group block reads (the "weight shuffle" layout).
//
// Computes  C[bm, n] = sum_k A[bm, k] * W[n, k]  for a Q6_K weight reordered at compile_model time
// into the SG-transposed ("shuffle") layout (see RepackGGUFWeightsShuffle). Group N-rows by
// OPG = SG_SIZE = 16; the OPG blocks of each K-block are interleaved in 4-byte chunks so one
// intel_sub_group_block_read delivers coalesced weight bytes to the 16 lanes.
//
//   pql_T:  total_blk * 128 bytes
//     Entry (h, bid) at (h*nbpr+bid)*OPG*128, sub-block j at j*OPG*16,
//     chunk c (0..3) at j*OPG*16 + c*OPG*4, lane lid gets 4 bytes.
//   pqh_T:  total_blk * 64 bytes
//     Entry at (h*nbpr+bid)*OPG*64, sub-block j at j*OPG*8, chunk c (0..1) at j*OPG*8 + c*OPG*4.
//   ps_T:   total_blk * 16 bytes   (int8 scales)
//     Entry at (h*nbpr+bid)*OPG*16, scale si (0..15) at si*OPG, 1 byte per lane.
//   pd_T:   total_blk * 2 bytes    (fp16 super-block scale)
//     Entry at (h*nbpr+bid)*OPG*2, 1 ushort per lane.
//
// Each lane owns ONE complete output row (running sum in a private register `acc`).
//
// == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
//   The baseline dispatch used ONE sub-group per work-group, so the number of concurrent hardware
//   threads == N/OPG. For small N this leaves the machine badly under-occupied, the dominant
//   bottleneck for this memory-bound kernel. We put KSPLIT sub-groups in each work-group (local =
//   OPG x 1 x KSPLIT). All KSPLIT sub-groups of a group cover the SAME OPG output rows but each owns
//   a strided subset of the K-blocks (sub-group sg handles bid = sg, sg+KSPLIT, ...). Each keeps a
//   private partial `acc`; at the end the KSPLIT partials are reduced through a small SLM buffer with
//   a single work-group barrier and sub-group 0 writes the output. Total live threads =
//   (N/OPG) * KSPLIT. KSPLIT=1 reproduces the original single-sub-group geometry exactly.
//
// The Q6_K reconstruction w = qval - 32 and the per-sub-block scale are factored OUT of the per-weight
// FMA: for sub-block j we compute  dot = sum a_k * qval_k  (qval in [0,63]) and  isum = sum a_k, then
// combine once as  acc += scale_j * (dot - 32*isum). The per-sub-group activation SLM is
// DOUBLE-BUFFERED so exactly one sub_group_barrier per K-block suffices.

// Accumulate one ql uint (4 positions, low+high nibble weights) into the two sub-block dot products
// and activation sums. Low-nibble weights use in[k], high-nibble use in[16+k].
#define Q6K_CHUNK(qlw, in, k0, s0, s1, s2, s3, qh_lo, qh_hi,                      \
                  dot_lo, dot_hi, isum_lo, isum_hi)                               \
    do {                                                                         \
        uint _q = (qlw);                                                         \
        uint _b0 =  _q        & 0xFFu;                                           \
        uint _b1 = (_q >>  8) & 0xFFu;                                           \
        uint _b2 = (_q >> 16) & 0xFFu;                                           \
        uint _b3 = (_q >> 24) & 0xFFu;                                           \
        float _l0 = (in)[(k0) + 0];  float _h0 = (in)[16 + (k0) + 0];            \
        float _l1 = (in)[(k0) + 1];  float _h1 = (in)[16 + (k0) + 1];            \
        float _l2 = (in)[(k0) + 2];  float _h2 = (in)[16 + (k0) + 2];            \
        float _l3 = (in)[(k0) + 3];  float _h3 = (in)[16 + (k0) + 3];            \
        (dot_lo) = fma(_l0, (float)((_b0 & 0xFu) | ((((qh_lo) >> (s0)) & 3u) << 4)), (dot_lo)); \
        (dot_lo) = fma(_l1, (float)((_b1 & 0xFu) | ((((qh_lo) >> (s1)) & 3u) << 4)), (dot_lo)); \
        (dot_lo) = fma(_l2, (float)((_b2 & 0xFu) | ((((qh_lo) >> (s2)) & 3u) << 4)), (dot_lo)); \
        (dot_lo) = fma(_l3, (float)((_b3 & 0xFu) | ((((qh_lo) >> (s3)) & 3u) << 4)), (dot_lo)); \
        (dot_hi) = fma(_h0, (float)(((_b0 >> 4) & 0xFu) | ((((qh_hi) >> (s0)) & 3u) << 4)), (dot_hi)); \
        (dot_hi) = fma(_h1, (float)(((_b1 >> 4) & 0xFu) | ((((qh_hi) >> (s1)) & 3u) << 4)), (dot_hi)); \
        (dot_hi) = fma(_h2, (float)(((_b2 >> 4) & 0xFu) | ((((qh_hi) >> (s2)) & 3u) << 4)), (dot_hi)); \
        (dot_hi) = fma(_h3, (float)(((_b3 >> 4) & 0xFu) | ((((qh_hi) >> (s3)) & 3u) << 4)), (dot_hi)); \
        (isum_lo) += _l0 + _l1 + _l2 + _l3;                                      \
        (isum_hi) += _h0 + _h1 + _h2 + _h3;                                      \
    } while (0)

// Process one full sub-block j (32 weights) given its 4 ql words + 2 qh words, combine with the
// pre-scaled clo/chi and accumulate into acc.
// Shift table {0,8,16,24, 2,10,18,26, 4,12,20,28, 6,14,22,30}.
#define Q6K_SUBBLOCK(in, ql0, ql1, ql2, ql3, qh_lo, qh_hi, clo, chi, acc)        \
    do {                                                                         \
        float _dl = 0.0f, _dh = 0.0f, _sl = 0.0f, _sh_ = 0.0f;                   \
        Q6K_CHUNK(ql0, in,  0,  0,  8, 16, 24, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        Q6K_CHUNK(ql1, in,  4,  2, 10, 18, 26, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        Q6K_CHUNK(ql2, in,  8,  4, 12, 20, 28, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        Q6K_CHUNK(ql3, in, 12,  6, 14, 22, 30, qh_lo, qh_hi, _dl, _dh, _sl, _sh_); \
        (acc) = fma((clo), _dl - 32.0f * _sl, (acc));                            \
        (acc) = fma((chi), _dh - 32.0f * _sh_, (acc));                           \
    } while (0)

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(fc_gguf_q6k_sg)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,       // activation [BM, K] (f16)
    const __global uchar*       W,       // shuffled weight buffer: [pql | pqh | ps | pd]
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
    const uint off_pql = 0u;
    const uint off_pqh = total_blk * 128u;
    const uint off_ps  = off_pqh + total_blk * 64u;
    const uint off_pd  = off_ps + total_blk * 16u;

    const __global uchar* PQL = W + off_pql;
    const __global uchar* PQH = W + off_pqh;
    const __global char*  PS  = (const __global char*)(W + off_ps);
    const __global uchar* PD  = W + off_pd;

    const __global uint* pql_u = (const __global uint*)PQL;
    const __global uint* pqh_u = (const __global uint*)PQH;

    const __global INPUT0_TYPE* A_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;

    // Each sub-group processes a strided subset of the K-blocks.
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;   // current buffer

        // --- Issue all long-latency global loads first (weights + scales) ---
        uint pql_e = (h * nbpr + bid) * (OPG * 128 / 4);   // uints
        uint8 w0 = BLOCK_READN(uint, 8, pql_u, pql_e +   0);  // sub 0,1
        uint8 w1 = BLOCK_READN(uint, 8, pql_u, pql_e + 128);  // sub 2,3
        uint8 w2 = BLOCK_READN(uint, 8, pql_u, pql_e + 256);  // sub 4,5
        uint8 w3 = BLOCK_READN(uint, 8, pql_u, pql_e + 384);  // sub 6,7

        uint pqh_e = (h * nbpr + bid) * (OPG * 64 / 4);    // uints
        uint8 hq0 = BLOCK_READN(uint, 8, pqh_u, pqh_e +   0);  // sub 0..3
        uint8 hq1 = BLOCK_READN(uint, 8, pqh_u, pqh_e + 128);  // sub 4..7

        uint ps_e = (h * nbpr + bid) * (OPG * 16);         // bytes
        uchar16 scq = BLOCK_READN(uchar, 16, (PS + ps_e), 0);

        uint pd_e = (h * nbpr + bid) * (OPG * 2);          // bytes
        ushort d_raw = BLOCK_READN(ushort, 1, (PD + pd_e), 0);

        // --- Load 256 activations into SLM (each lane writes 16 elements at stride OPG) ---
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < 16u; ++i) {
            sb[i * OPG + hh] = (float)A_row[bid * 256u + i * OPG + hh];
        }

        // --- Decode d and the 8 (clo, chi) scale pairs (overlaps outstanding loads) ---
        float d = (float)as_half(d_raw);
        char sc[16];
        sc[ 0]=as_char(scq.s0); sc[ 1]=as_char(scq.s1); sc[ 2]=as_char(scq.s2); sc[ 3]=as_char(scq.s3);
        sc[ 4]=as_char(scq.s4); sc[ 5]=as_char(scq.s5); sc[ 6]=as_char(scq.s6); sc[ 7]=as_char(scq.s7);
        sc[ 8]=as_char(scq.s8); sc[ 9]=as_char(scq.s9); sc[10]=as_char(scq.sa); sc[11]=as_char(scq.sb);
        sc[12]=as_char(scq.sc); sc[13]=as_char(scq.sd); sc[14]=as_char(scq.se); sc[15]=as_char(scq.sf);

        float clo[8], chi[8];
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < 8; ++j) {
            clo[j] = d * (float)sc[2*j];
            chi[j] = d * (float)sc[2*j + 1];
        }

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // --- 8 sub-blocks: dot products + factored scale/bias combine ---
        __local const float* in;

        in = sb + 0*32; Q6K_SUBBLOCK(in, w0.s0, w0.s1, w0.s2, w0.s3, hq0.s0, hq0.s1, clo[0], chi[0], acc);
        in = sb + 1*32; Q6K_SUBBLOCK(in, w0.s4, w0.s5, w0.s6, w0.s7, hq0.s2, hq0.s3, clo[1], chi[1], acc);
        in = sb + 2*32; Q6K_SUBBLOCK(in, w1.s0, w1.s1, w1.s2, w1.s3, hq0.s4, hq0.s5, clo[2], chi[2], acc);
        in = sb + 3*32; Q6K_SUBBLOCK(in, w1.s4, w1.s5, w1.s6, w1.s7, hq0.s6, hq0.s7, clo[3], chi[3], acc);
        in = sb + 4*32; Q6K_SUBBLOCK(in, w2.s0, w2.s1, w2.s2, w2.s3, hq1.s0, hq1.s1, clo[4], chi[4], acc);
        in = sb + 5*32; Q6K_SUBBLOCK(in, w2.s4, w2.s5, w2.s6, w2.s7, hq1.s2, hq1.s3, clo[5], chi[5], acc);
        in = sb + 6*32; Q6K_SUBBLOCK(in, w3.s0, w3.s1, w3.s2, w3.s3, hq1.s4, hq1.s5, clo[6], chi[6], acc);
        in = sb + 7*32; Q6K_SUBBLOCK(in, w3.s4, w3.s5, w3.s6, w3.s7, hq1.s6, hq1.s7, clo[7], chi[7], acc);
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

#elif GG_FMT_Q8_0
// ==================================================================================================
// High-performance Q8_0 GEMV with Intel sub-group block reads (the "weight shuffle" layout).
//
// Computes  C[bm, n] = sum_k A[bm, k] * W[n, k]  for a Q8_0 weight that has been reordered at
// compile_model time into the SG-transposed ("shuffle") layout (see RepackGGUFWeightsShuffle).
//
// == Q8_0 format recap ==
//   GGUF block_q8_0 = { half d; int8_t qs[32]; }  (34 bytes, QK=32 weights)
//     dequant:  w = q * d          (q is signed int8, no zero-point / min)
//
//   EIGHT consecutive Q8_0 blocks are grouped into one super-block of 256 weights. Each super-block
//   carries:
//     * 8 sub-blocks x 32 qs bytes            = 256 bytes  (weights,  "pqs")
//     * 8 fp16 scales d                       =  16 bytes  (scales,   "pd")
//   which equals the original 8 x 34 = 272 bytes — a pure reorder.
//
//   Because w = q * d and q is signed, there is NO min/bias term:
//       acc += d_j * dot_j          dot_j = sum_i x_i * q_i   (32 weights)
//
// == Combined weight buffer (SAME total size as the native weight) ==
//   W = [ pqs | pd ]
//   pqs:  total_blk * 256 bytes at offset 0.
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 256 bytes (4096 B)
//     Sub-block j at j*OPG*32, chunk c (0..7) at j*OPG*32 + c*OPG*4.
//     One intel_sub_group_block_read8 per sub-block delivers 8 int8x4 chunks (32 signed weights).
//   pd:   total_blk * 16 bytes at offset total_blk*256.
//     Entry (h, bid) at (h*nbpr + bid) * OPG * 16 bytes (256 B), SoA fp16 field j at j*OPG*2 + lid*2.
//     One intel_sub_group_block_read_us8 fetches all 8 scales per lane.
//
// == Occupancy optimization: K-split across sub-groups (KSPLIT) ==
//   Same technique as the Q4_K section: KSPLIT sub-groups per work-group cover the SAME OPG output
//   rows but each owns a strided subset of the K-blocks; a single work-group barrier at the end
//   reduces the partials. KSPLIT=1 reproduces the original single-sub-group geometry exactly.

// Accumulate one 4-byte chunk (4 signed int8 weights) into the sub-block dot product. `in` points at
// slm base of the sub-block, `c` is the chunk index (0..7); weights c*4..c*4+3 map to activations
// in[c*4..c*4+3].
#define ACC_CHUNK_Q8(dot, u, c, in)                                              \
    do {                                                                         \
        char4 _q = as_char4((uint)(u));                                          \
        (dot) = fma((in)[(c)*4 + 0], (float)_q.s0, (dot));                       \
        (dot) = fma((in)[(c)*4 + 1], (float)_q.s1, (dot));                       \
        (dot) = fma((in)[(c)*4 + 2], (float)_q.s2, (dot));                       \
        (dot) = fma((in)[(c)*4 + 3], (float)_q.s3, (dot));                       \
    } while (0)

// Process one full sub-block j (32 signed weights = 8 chunks).
#define Q8_SUBBLOCK(in, wv, sc, acc)                                             \
    do {                                                                         \
        float _dot = 0.0f;                                                       \
        ACC_CHUNK_Q8(_dot, (wv).s0, 0, in); ACC_CHUNK_Q8(_dot, (wv).s1, 1, in);  \
        ACC_CHUNK_Q8(_dot, (wv).s2, 2, in); ACC_CHUNK_Q8(_dot, (wv).s3, 3, in);  \
        ACC_CHUNK_Q8(_dot, (wv).s4, 4, in); ACC_CHUNK_Q8(_dot, (wv).s5, 5, in);  \
        ACC_CHUNK_Q8(_dot, (wv).s6, 6, in); ACC_CHUNK_Q8(_dot, (wv).s7, 7, in);  \
        (acc) = fma((sc), _dot, (acc));                                          \
    } while (0)

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(fc_gguf_q8_0_sg)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,       // activation [BM, K] (f16)
    const __global uchar*       W,       // shuffled weight buffer: [pqs | pd]
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
    const uint off_pd  = total_blk * 256u;

    const __global uint*   pqs_u = (const __global uint*)(W + off_pqs);
    const __global ushort* pd_us = (const __global ushort*)(W + off_pd);

    const __global INPUT0_TYPE* A_row = A + (uint)bm * (uint)K_SIZE;

    float acc = 0.0f;

    // Each sub-group processes a strided subset of the K-blocks.
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;   // current buffer

        // --- Issue all long-latency global loads first (weights + scales). Each sub-block is
        //     OPG*32 bytes = OPG*8 uints apart. ---
        uint pe_u = (h * nbpr + bid) * (OPG * 256 / 4);   // uints
        uint8 w0 = BLOCK_READN(uint, 8, pqs_u, pe_u + 0*OPG*8);  // sub 0
        uint8 w1 = BLOCK_READN(uint, 8, pqs_u, pe_u + 1*OPG*8);  // sub 1
        uint8 w2 = BLOCK_READN(uint, 8, pqs_u, pe_u + 2*OPG*8);  // sub 2
        uint8 w3 = BLOCK_READN(uint, 8, pqs_u, pe_u + 3*OPG*8);  // sub 3
        uint8 w4 = BLOCK_READN(uint, 8, pqs_u, pe_u + 4*OPG*8);  // sub 4
        uint8 w5 = BLOCK_READN(uint, 8, pqs_u, pe_u + 5*OPG*8);  // sub 5
        uint8 w6 = BLOCK_READN(uint, 8, pqs_u, pe_u + 6*OPG*8);  // sub 6
        uint8 w7 = BLOCK_READN(uint, 8, pqs_u, pe_u + 7*OPG*8);  // sub 7

        // 8 fp16 scales per lane (SoA), fetched with ONE wide block read.
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    // ushorts
        ushort8 dv = BLOCK_READN(ushort, 8, pd_us, pd_off);

        // --- Load 256 activations into SLM (each lane writes 16 elements at stride OPG) ---
        __attribute__((opencl_unroll_hint))
        for (uint i = 0; i < 16u; ++i) {
            sb[i * OPG + hh] = (float)A_row[bid * 256u + i * OPG + hh];
        }

        // --- Decode 8 fp16 scales (overlaps outstanding loads) ---
        float scale[8];
        {
            ushort dh[8];
            dh[0]=dv.s0; dh[1]=dv.s1; dh[2]=dv.s2; dh[3]=dv.s3;
            dh[4]=dv.s4; dh[5]=dv.s5; dh[6]=dv.s6; dh[7]=dv.s7;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < 8; ++i)
                scale[i] = (float)as_half(dh[i]);
        }

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        // --- 8 sub-blocks: dot products + per-sub-block scale ---
        __local const float* in;
        in = sb + 0*32; Q8_SUBBLOCK(in, w0, scale[0], acc);
        in = sb + 1*32; Q8_SUBBLOCK(in, w1, scale[1], acc);
        in = sb + 2*32; Q8_SUBBLOCK(in, w2, scale[2], acc);
        in = sb + 3*32; Q8_SUBBLOCK(in, w3, scale[3], acc);
        in = sb + 4*32; Q8_SUBBLOCK(in, w4, scale[4], acc);
        in = sb + 5*32; Q8_SUBBLOCK(in, w5, scale[5], acc);
        in = sb + 6*32; Q8_SUBBLOCK(in, w6, scale[6], acc);
        in = sb + 7*32; Q8_SUBBLOCK(in, w7, scale[7], acc);
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

#else
#error "fc_gguf_gemv_opt.cl: define exactly one GG_FMT_* format macro"
#endif
