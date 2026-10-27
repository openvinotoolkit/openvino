// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u + 0*OPG*8);  // sub 0
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 1*OPG*8);  // sub 1
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 2*OPG*8);  // sub 2
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 3*OPG*8);  // sub 3
        uint8 w4 = intel_sub_group_block_read8(pqs_u + pe_u + 4*OPG*8);  // sub 4
        uint8 w5 = intel_sub_group_block_read8(pqs_u + pe_u + 5*OPG*8);  // sub 5
        uint8 w6 = intel_sub_group_block_read8(pqs_u + pe_u + 6*OPG*8);  // sub 6
        uint8 w7 = intel_sub_group_block_read8(pqs_u + pe_u + 7*OPG*8);  // sub 7

        // 8 fp16 scales per lane (SoA), fetched with ONE wide block read.
        uint    pd_off = (h * nbpr + bid) * (OPG * 8);    // ushorts
        ushort8 dv = intel_sub_group_block_read_us8(pd_us + pd_off);

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
