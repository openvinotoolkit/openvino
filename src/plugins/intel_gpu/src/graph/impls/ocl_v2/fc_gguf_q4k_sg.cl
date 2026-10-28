// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
// `in` points at slm base of the sub-block, `c` is the chunk index (0..3). Low nibbles map to
// in[c*4 + b], high nibbles to in[16 + c*4 + b].
#define Q4K_ACC_CHUNK(dot, u, c, in)                                             \
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
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);  // sub 0,1
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);  // sub 2,3
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);  // sub 4,5
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);  // sub 6,7

        uint   psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);   // uints
        uint   psl_us_off = (h * nbpr + bid) * (OPG * 16 / 2);   // ushort
        uint   sl     = intel_sub_group_block_read   (psl_pu + psl_pu_off +  0);
        uint   ml     = intel_sub_group_block_read   (psl_pu + psl_pu_off + 16);
        ushort sh     = intel_sub_group_block_read_us(psl_us + psl_us_off + 64);
        ushort mh     = intel_sub_group_block_read_us(psl_us + psl_us_off + 80);
        ushort d_raw  = intel_sub_group_block_read_us(psl_us + psl_us_off + 96);
        ushort dn_raw = intel_sub_group_block_read_us(psl_us + psl_us_off + 112);

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
        Q4K_ACC_CHUNK(dot, w0.s0, 0, in); Q4K_ACC_CHUNK(dot, w0.s1, 1, in);
        Q4K_ACC_CHUNK(dot, w0.s2, 2, in); Q4K_ACC_CHUNK(dot, w0.s3, 3, in);
        acc = fma(scale[0], dot, acc); acc = fma(-minv[0], isum[0], acc);

        in = sb + 1*32; dot = 0.0f;
        Q4K_ACC_CHUNK(dot, w0.s4, 0, in); Q4K_ACC_CHUNK(dot, w0.s5, 1, in);
        Q4K_ACC_CHUNK(dot, w0.s6, 2, in); Q4K_ACC_CHUNK(dot, w0.s7, 3, in);
        acc = fma(scale[1], dot, acc); acc = fma(-minv[1], isum[1], acc);

        in = sb + 2*32; dot = 0.0f;
        Q4K_ACC_CHUNK(dot, w1.s0, 0, in); Q4K_ACC_CHUNK(dot, w1.s1, 1, in);
        Q4K_ACC_CHUNK(dot, w1.s2, 2, in); Q4K_ACC_CHUNK(dot, w1.s3, 3, in);
        acc = fma(scale[2], dot, acc); acc = fma(-minv[2], isum[2], acc);

        in = sb + 3*32; dot = 0.0f;
        Q4K_ACC_CHUNK(dot, w1.s4, 0, in); Q4K_ACC_CHUNK(dot, w1.s5, 1, in);
        Q4K_ACC_CHUNK(dot, w1.s6, 2, in); Q4K_ACC_CHUNK(dot, w1.s7, 3, in);
        acc = fma(scale[3], dot, acc); acc = fma(-minv[3], isum[3], acc);

        in = sb + 4*32; dot = 0.0f;
        Q4K_ACC_CHUNK(dot, w2.s0, 0, in); Q4K_ACC_CHUNK(dot, w2.s1, 1, in);
        Q4K_ACC_CHUNK(dot, w2.s2, 2, in); Q4K_ACC_CHUNK(dot, w2.s3, 3, in);
        acc = fma(scale[4], dot, acc); acc = fma(-minv[4], isum[4], acc);

        in = sb + 5*32; dot = 0.0f;
        Q4K_ACC_CHUNK(dot, w2.s4, 0, in); Q4K_ACC_CHUNK(dot, w2.s5, 1, in);
        Q4K_ACC_CHUNK(dot, w2.s6, 2, in); Q4K_ACC_CHUNK(dot, w2.s7, 3, in);
        acc = fma(scale[5], dot, acc); acc = fma(-minv[5], isum[5], acc);

        in = sb + 6*32; dot = 0.0f;
        Q4K_ACC_CHUNK(dot, w3.s0, 0, in); Q4K_ACC_CHUNK(dot, w3.s1, 1, in);
        Q4K_ACC_CHUNK(dot, w3.s2, 2, in); Q4K_ACC_CHUNK(dot, w3.s3, 3, in);
        acc = fma(scale[6], dot, acc); acc = fma(-minv[6], isum[6], acc);

        in = sb + 7*32; dot = 0.0f;
        Q4K_ACC_CHUNK(dot, w3.s4, 0, in); Q4K_ACC_CHUNK(dot, w3.s5, 1, in);
        Q4K_ACC_CHUNK(dot, w3.s6, 2, in); Q4K_ACC_CHUNK(dot, w3.s7, 3, in);
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
