// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#pragma OPENCL EXTENSION cl_intel_subgroups        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short  : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char   : enable

#ifndef OPG
#define OPG 16  // = SG_SIZE = sub_group_size
#endif

#ifndef KSPLIT
#define KSPLIT 1  // sub-groups per work-group (K-split factor); set from CPP JIT
#endif

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
        uint8 w0 = intel_sub_group_block_read8(pql_u + pql_e +   0);  // sub 0,1
        uint8 w1 = intel_sub_group_block_read8(pql_u + pql_e + 128);  // sub 2,3
        uint8 w2 = intel_sub_group_block_read8(pql_u + pql_e + 256);  // sub 4,5
        uint8 w3 = intel_sub_group_block_read8(pql_u + pql_e + 384);  // sub 6,7

        uint pqh_e = (h * nbpr + bid) * (OPG * 64 / 4);    // uints
        uint8 hq0 = intel_sub_group_block_read8(pqh_u + pqh_e +   0);  // sub 0..3
        uint8 hq1 = intel_sub_group_block_read8(pqh_u + pqh_e + 128);  // sub 4..7

        uint ps_e = (h * nbpr + bid) * (OPG * 16);         // bytes
        uchar16 scq = intel_sub_group_block_read_uc16((const __global uchar*)(PS + ps_e));

        uint pd_e = (h * nbpr + bid) * (OPG * 2);          // bytes
        ushort d_raw = intel_sub_group_block_read_us((const __global ushort*)(PD + pd_e));

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
