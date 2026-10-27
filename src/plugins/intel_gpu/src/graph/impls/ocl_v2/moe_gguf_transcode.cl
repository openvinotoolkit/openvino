// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GGUF MoE weight transcode kernel (compute-bound / prefill path, token > 32).
//
// Converts one projection's raw GGUF block-quantised expert weights W[E, N, K] into a
// OneDNN-WOQ-native low-bit layout consumed by the existing per-expert oneDNN prefill loop
// (exec_prefill_onednn / init_dnnl_weights):
//   - a packed weight scratchpad WQ: i4 (TRANSCODE_TO_I4=1) or i8 (signed), physical [E, N, K]
//     (per expert matches dnnl wei_md [K,N] with format_tag::ba), and
//   - a parallel f16 per-group scale scratchpad SC[E, K/REQUANT_GROUP, N]: element (e,g,n) at
//     (e*(K/REQUANT_GROUP) + g)*N + n  (per-expert x per-K-group x per-N == dnnl scale md {K/gs,N} ab).
//
// This is the multi-expert fork of fc_gguf_transcode.cl: identical per-format block decoders (so the
// numerics track the native GEMV/decode kernel exactly), plus an expert axis on global_id(2). The
// decoded f16 values live ONLY in registers and are symmetrically re-quantised per REQUANT_GROUP into
// the low-bit domain — dequant NEVER lands in an f16/f32 weight buffer (constraint MC2). The only
// persisted weight is the low-bit scratchpad, which the caller recycles (shared per-stream arena, MC5).
//
// One work-item owns one (expert e, output row n, GGUF block blk):
//   global = [N_SIZE, K_SIZE / GGUF_BLOCK_ELEM, NUM_EXPERTS], local = [SG, 1, 1].
// REQUANT_GROUP must divide GGUF_BLOCK_ELEM (a group never straddles two GGUF blocks).
//
// ----------------------------------------------------------------------------------------------
// GGUF_SG_LAYOUT (Q4_K / Q5_K / Q6_K only): the routed/shared weight Constant for this projection
// was repacked at graph-compile time (RepackGGUFMoEWeights::pack_moe_weight_sg, see
// repack_gguf_moe_weights.cpp) into the transposed "SG" (sub-group block-read) layout consumed by
// the decode-path GEMV kernels (moe_gguf_sg_gemv.cl), instead of the plain per-row raw GGUF block
// layout this file was originally written against. When GGUF_SG_LAYOUT is set, Stage 1 below
// gathers this work-item's block bytes from the SG-packed sections (qs/ql, qh, psl/ps, pd) instead
// of indexing a contiguous per-row block, and decodes directly from that "shuffled" representation
// (mtq_decode_block_sg); the requantize stage (Stage 2) is unchanged. See moe_gguf_sg_gemv.cl's
// header comment for the full byte layout. This kernel is always dispatched with
// local = (SG_SIZE=16, 1, 1) and SG_SIZE == the SG pack's OPG(16), so get_group_id(0) == the
// "row-group" index h and get_local_id(0) == the "lane" lid directly (no extra div/mod needed).
//
// Every SG-packed section is laid out, per 16-row group, as SG_OPG(=16) consecutive lanes' bytes
// interleaved in 4-byte (or smaller) chunks: chunk c of the section lives at
// `section_base + c*SG_OPG*4 + lid*4` (uint chunks) or `... + lid*2` (ushort fields) etc. That is
// EXACTLY the layout intel_sub_group_block_read(2/4/8)/_us(8)/_uc16 are built to consume: one
// hardware block-load transaction delivers chunk (c0+i) to lane i for the whole sub-group, instead
// of SG_OPG independent scalar loads per chunk that the compiler cannot always coalesce/vectorize
// on its own. The decoders below therefore gather each section with explicit sub-group block reads
// (mtq_gather_plane / direct intel_sub_group_block_read* calls) into small private arrays, then
// decode from those private bytes — mirroring the coalesced-gather idiom fc_gguf_transcode.cl's
// GGUF_SHUFFLE path uses (sh_gather_plane) for the single-expert FC prefill. This is purely an
// addressing/scheduling change: the decoded values and their derivation are bit-exact with the
// original per-lane scalar-pointer version (same bytes, same arithmetic, only fetched via a single
// wide sub-group transaction instead of N independent global loads).
// ----------------------------------------------------------------------------------------------

// GGUF MoE weight transcode kernel — Xe3-LPG tuned v5.
//
// Builds on v4 (6.975 ms). VTune showed SBID (67-70%) + Send (60-66%) stalls with
// ALU1 (INT) at 70-75% of instructions — the round/clamp/bit-pack chain is the
// dependency bottleneck. v5 changes:
//   * Vectorize the requant multiply + round + clamp over float8/int8 so the
//     compiler emits packed SIMD ALU ops (fewer instructions, shorter dep chains),
//     directly attacking the SBID/Dist-Acc stalls.
//   * Keep v4's fused amax + single (float)half materialization (kills scratch reload).
//   * Keep 16-wide sub-group for a clean 32B coalesced SC store and wide packed WQ stores.
//   * GGUF_SG_LAYOUT decode (Stage 1) now gathers each SG-packed section with explicit
//     intel_sub_group_block_read* transactions instead of per-lane scalar-pointer loads
//     (see header comment above) — same win fc_gguf_transcode.cl's shuffle path banks on:
//     one wide Send per section instead of SG_OPG scattered scalar Sends, which is where
//     the "Send 60-66%" stall time was going for the decode-path Q4_K/Q5_K/Q6_K/Q8_0 experts.
//
// GGUF_SG_LAYOUT — v6 regression fix (measured on real HW, see below).
//
// v5's GGUF_SG_LAYOUT gather rework made the *decode-path* GEMV kernel faster but was reported to
// make THIS (compute-bound, high-occupancy) transcode kernel slower for Q4_K/Q5_K, and only a
// partial win for Q6_K. Standalone microbenchmarking against the real dispatch shapes confirmed
// this (Arc B580, values are median of 20 iters, transcode kernel only):
//     Q4_K E256 N512  K2048: raw(no-SG)=2.1ms   SG(v5)=4.1ms   (~90% SLOWER)
//     Q4_K E256 N2048 K512 : raw(no-SG)=3.8ms   SG(v5)=4.1ms   (~7%  slower)
//     Q5_K E256 N2048 K512 : raw(no-SG)=4.7ms   SG(v5)=5.2ms   (~11% slower)
//     Q6_K E256 N512  K2048: raw(no-SG)=10.7ms  SG(v5)=6.0ms   (~44% faster)
// CL_KERNEL_PRIVATE_MEM_SIZE showed the SG decoders spilling private registers (Q4_K: +2KB,
// Q5_K: +2.5KB, Q6_K: +3.3KB over the Stage-2 baseline) that a decode-only (no Stage-2) build did
// NOT spill — i.e. the gather/fetch itself (mtq_gather_plane, the per-lane block reads) was never
// the bottleneck; it was mtq_decode_block_sg's per-group unpack loop
// (`for (int j = 0; j < 8; ++j) { ... variable-shift bit-extraction ... }`, shared verbatim by the
// Q4_K/Q5_K/Q6_K SG decoders) staying a REAL runtime loop with per-lane-uniform but
// loop-variable-dependent shift amounts (`sl >> (j*4)`, `sh >> (j*2)`, ...). Because j was not
// compile-time constant, the compiler could not fold those into fixed shifts and kept
// re-deriving/re-spilling the freshly block-read section values (sl/ml/sh/mh/qs_bytes/...) every
// iteration instead of holding them live across all 8 unrolled group-decodes.
// Fix (both changes below, no numerics affected — same bytes, same arithmetic, verified bit-exact):
//   1. Add __attribute__((opencl_unroll_hint)) to that `for (j = 0; j < 8; ++j)` loop (all three
//      of Q4_K/Q5_K/Q6_K's mtq_decode_block_sg). This alone let the compiler constant-fold the
//      per-j shift amounts and eliminated the private-mem spill entirely.
//   2. Q4_K/Q5_K also had 6 separate 1-Send-each intel_sub_group_block_read/_us calls for the tiny
//      psl SoA fields (sl/ml/sh/mh/d/dmin); sl+ml are two adjacent 4B/lane chunks and
//      sh+mh+d+dmin are four adjacent 2B/lane chunks, so intel_sub_group_block_read2 /
//      intel_sub_group_block_read_us4 fetch the same bytes in 2 Sends instead of 6
//      (block_read2/read_us4 are defined as the exact per-lane concatenation of the equivalent
//      separate block_read calls — Send-count-only change, bit-exact).
// Result (same benchmark): Q4_K SG ~2.83-2.85ms (was 4.1ms, now within ~2-30% of / faster than
// raw), Q5_K SG ~4.3-4.8ms (was 5.2-5.8ms, now on par with raw), Q6_K SG ~4.6-4.8ms (was 6.0ms,
// now clearly faster than raw, up from +44% to +about 2x). GGUF_SG_LAYOUT stays enabled for all
// three (still a net win, or at worst competitive) — no dispatch/host-side change needed.

#include "include/batch_headers/common.cl"

inline half FUNC(mtq_load_f16)(const __global uchar* p) {
    ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    return as_half(bits);
}

#if defined(GGUF_IS_Q4_0)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(mtq_load_f16)(blk);
    const __global uchar* qs = blk + 2;
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 16; ++j) {
        const uchar b = qs[j];
        out[j]      = (half)(((int)(b & 0x0F) - 8) * (float)d);
        out[j + 16] = (half)(((int)(b >> 4)   - 8) * (float)d);
    }
}
#endif

#if defined(GGUF_IS_Q8_0)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const half d = FUNC_CALL(mtq_load_f16)(blk);
    const __global char* qs = (const __global char*)(blk + 2);
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 32; ++j) out[j] = (half)((float)qs[j] * (float)d);
}
#endif

#if defined(GGUF_IS_Q4_K) || defined(GGUF_IS_Q5_K)
inline void FUNC(mtq_scale_min_k4)(int j, const __global uchar* q, uchar* d, uchar* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (uchar)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = (uchar)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}
#endif

#if defined(GGUF_IS_Q4_K)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(mtq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(mtq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qs     = blk + 16;
    int o = 0, is = 0;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(mtq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(mtq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d1 * (float)(qs[l] & 0x0F) - m1);
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) out[o++] = (half)(d2 * (float)(qs[l] >> 4) - m2);
        qs += 32; is += 2;
    }
}
#endif

#if defined(GGUF_IS_Q5_K)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const float d    = (float)FUNC_CALL(mtq_load_f16)(blk);
    const float dmin = (float)FUNC_CALL(mtq_load_f16)(blk + 2);
    const __global uchar* scales = blk + 4;
    const __global uchar* qh     = blk + 16;
    const __global uchar* ql     = blk + 48;
    int o = 0, is = 0; uchar u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uchar sc, m;
        FUNC_CALL(mtq_scale_min_k4)(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = dmin * m;
        FUNC_CALL(mtq_scale_min_k4)(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = dmin * m;
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0); out[o++] = (half)(d1 * (float)q - m1); }
        __attribute__((opencl_unroll_hint))
        for (int l = 0; l < 32; ++l) { const int q = (int)(ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0); out[o++] = (half)(d2 * (float)q - m2); }
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
#endif

#if defined(GGUF_IS_Q6_K)
inline void FUNC(mtq_decode_block)(const __global uchar* blk, __private half* out) {
    const __global uchar* ql = blk;
    const __global uchar* qh = blk + 128;
    const __global char*  sc = (const __global char*)(blk + 192);
    const float d = (float)FUNC_CALL(mtq_load_f16)(blk + 208);
    int o = 0;
    for (int n = 0; n < 256; n += 128) {
        __attribute__((opencl_unroll_hint))
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

// ================================================================================================
// GGUF_SG_LAYOUT decoders: decode ONE GGUF super-block directly from the transposed "SG" per-expert
// layout (moe_gguf_sg_gemv.cl / repack_gguf_moe_weights.cpp pack_expert_sg), for row n = h*16+lid,
// block bid. Numerically identical to the plain mtq_decode_block above (verified byte/value-exact
// against the raw-GGUF-block reference offline); only the *addressing/fetch strategy* differs,
// because the SG pack interleaves 16 consecutive rows' bytes for coalesced sub-group reads.
//
// `w_expert_base` points at the start of THIS expert's SG-packed bytes (same total per-expert byte
// size as the raw layout). Section layout (matching pack_expert_sg): [ qs/ql | qh(Q5_K/Q6_K only) |
// psl/ps | pd(Q6_K only) ], each section sized num_blocks_total * <section_bytes_per_block>, where
// num_blocks_total = N_SIZE * blocks_per_row (this projection's per-expert block count).
//
// Fetch strategy: every field access below happens at a sub-group-uniform base address (identical
// across all 16 lanes) with the per-lane offset implied by the intel_sub_group_block_read* lane
// mapping — i.e. exactly the pattern these intrinsics are designed for. mtq_gather_plane() pulls a
// whole 4-byte-chunked section into private memory with one wide block read per 8 chunks (matching
// fc_gguf_transcode.cl's sh_gather_plane for the single-expert FC path); the smaller SoA scalar
// fields (sl/ml/sh/mh/d/dmin, per-lane scale bytes, per-lane f16 scales) use single block reads of
// the matching width (block_read/_us/_uc16). This turns what used to be up to 32 independent scalar
// global loads per lane (SG_OPG=16x redundant Send traffic for the *same* transaction shape) into a
// small, fixed number of sub-group-wide block-read Sends per decoded GGUF block.
// ================================================================================================
#if GGUF_SG_LAYOUT

#pragma OPENCL EXTENSION cl_intel_subgroups       : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char  : enable

#define SG_OPG 16

// Coalesced sub-group block gather of a 4-byte-chunked SG-packed section into private uints.
// `base` must be the section's per-group pointer (already offset by group_idx); `entry` is an
// additional byte offset (kept for symmetry with fc_gguf_transcode.cl's sh_gather_plane, always 0
// here since every caller passes an already-offset group pointer). Chunk c of the section lives at
// `base + entry + c*SG_OPG*4`, spanning SG_OPG consecutive uints — exactly what
// intel_sub_group_block_read(8) delivers to the SG_OPG lanes in one transaction. Requires the
// launch sub-group width == SG_OPG, guaranteed by this kernel's intel_reqd_sub_group_size(16).
inline void FUNC(mtq_gather_plane)(const __global uchar* base, uint entry, __private uint* out_u, int n_uint) {
    const __global uint* p = (const __global uint*)(base + entry);
    int c = 0;
    // Wide 8-chunk (512 B) block reads to keep many loads in flight.
    __attribute__((opencl_unroll_hint))
    for (; c + 8 <= n_uint; c += 8) {
        const uint8 v = intel_sub_group_block_read8(p + (uint)c * (uint)SG_OPG);
        out_u[c + 0] = v.s0; out_u[c + 1] = v.s1; out_u[c + 2] = v.s2; out_u[c + 3] = v.s3;
        out_u[c + 4] = v.s4; out_u[c + 5] = v.s5; out_u[c + 6] = v.s6; out_u[c + 7] = v.s7;
    }
    // Tail (n_uint not a multiple of 8): one chunk per block read.
    __attribute__((opencl_unroll_hint))
    for (; c < n_uint; ++c) {
        out_u[c] = intel_sub_group_block_read(p + (uint)c * (uint)SG_OPG);
    }
}

#if defined(GGUF_IS_Q4_K)
inline void FUNC(mtq_decode_block_sg)(const __global uchar* w_expert_base, int h, int lid, int bid,
                                      int blocks_per_row, __private half* out) {
    const uint num_blocks = (uint)N_SIZE * (uint)blocks_per_row;
    const __global uchar* qs_sec  = w_expert_base;
    const __global uchar* psl_sec = qs_sec + (size_t)num_blocks * 128;

    const uint group_idx = (uint)h * (uint)blocks_per_row + (uint)bid;
    const __global uchar* qs_group  = qs_sec  + (size_t)group_idx * SG_OPG * 128;
    const __global uchar* psl_group = psl_sec + (size_t)group_idx * SG_OPG * 16;

    // qs: 128 B/lane = 32 uint chunks -> one gather (4 wide block_read8 transactions).
    __private uint qs_u[32];
    FUNC_CALL(mtq_gather_plane)(qs_group, 0u, qs_u, 32);
    const __private uchar* qs_bytes = (const __private uchar*)qs_u;

    // psl SoA (per-lane, coalesced): sl@lid*4 ml@64+lid*4 sh@128+lid*2 mh@160+lid*2 d@192+lid*2 dmin@224+lid*2.
    // sl/ml (2 adjacent 4B/lane chunks) and sh/mh/d/dmin (4 adjacent 2B/lane chunks) are each
    // fetched with ONE wide block read instead of one Send per scalar field (6 Sends -> 2); see
    // "v6" header note below. block_read2/read_us4 are defined as the exact per-lane concatenation
    // of the equivalent separate block_read/block_read_us calls, so this is bit-exact, Send-count-only.
    const __global uint*   psl_u = (const __global uint*)psl_group;
    const __global ushort* psl_s = (const __global ushort*)psl_group;
    const uint2   sl_ml         = intel_sub_group_block_read2   (psl_u + 0u);   // sl@0, ml@64B
    const ushort4 sh_mh_d_dmin  = intel_sub_group_block_read_us4(psl_s + 64u);  // sh@128B mh@160B d@192B dmin@224B
    const uint   sl = sl_ml.s0;
    const uint   ml = sl_ml.s1;
    const ushort sh = sh_mh_d_dmin.s0;
    const ushort mh = sh_mh_d_dmin.s1;
    const ushort dd = sh_mh_d_dmin.s2;
    const ushort dn = sh_mh_d_dmin.s3;
    const float d    = (float)as_half(dd);
    const float dmin = (float)as_half(dn);

    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 8; ++j) {
        const uint sq = ((sl >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
        const uint mq = ((ml >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
        const float scale = (float)sq * d, minv = (float)mq * dmin;
        __attribute__((opencl_unroll_hint))
        for (int kc = 0; kc < 4; ++kc) {
            __attribute__((opencl_unroll_hint))
            for (int kb = 0; kb < 4; ++kb) {
                const int b = kc * 4 + kb;
                const uchar byte = qs_bytes[j * 16 + b];
                out[j * 32 + b]      = (half)((float)(byte & 0x0F) * scale - minv);
                out[j * 32 + 16 + b] = (half)((float)((byte >> 4) & 0x0F) * scale - minv);
            }
        }
    }
}
#endif

#if defined(GGUF_IS_Q5_K)
inline void FUNC(mtq_decode_block_sg)(const __global uchar* w_expert_base, int h, int lid, int bid,
                                      int blocks_per_row, __private half* out) {
    const uint num_blocks = (uint)N_SIZE * (uint)blocks_per_row;
    const __global uchar* qs_sec  = w_expert_base;
    const __global uchar* qh_sec  = qs_sec + (size_t)num_blocks * 128;
    const __global uchar* psl_sec = qh_sec + (size_t)num_blocks * 32;

    const uint group_idx = (uint)h * (uint)blocks_per_row + (uint)bid;
    const __global uchar* qs_group  = qs_sec  + (size_t)group_idx * SG_OPG * 128;
    const __global uchar* qh_group  = qh_sec  + (size_t)group_idx * SG_OPG * 32;
    const __global uchar* psl_group = psl_sec + (size_t)group_idx * SG_OPG * 16;

    // qs: 128 B/lane = 32 uint chunks; qh: 32 B/lane = 8 uint chunks. One gather each.
    __private uint qs_u[32];
    FUNC_CALL(mtq_gather_plane)(qs_group, 0u, qs_u, 32);
    const __private uchar* qs_bytes = (const __private uchar*)qs_u;

    __private uint qh_u[8];
    FUNC_CALL(mtq_gather_plane)(qh_group, 0u, qh_u, 8);
    const __private uchar* qh_bytes = (const __private uchar*)qh_u;

    // psl SoA (per-lane, coalesced): sl@lid*4 ml@64+lid*4 sh@128+lid*2 mh@160+lid*2 d@192+lid*2 dmin@224+lid*2.
    // Consolidated fetch (see Q4_K's mtq_decode_block_sg comment above): 6 Sends -> 2.
    const __global uint*   psl_u = (const __global uint*)psl_group;
    const __global ushort* psl_s = (const __global ushort*)psl_group;
    const uint2   sl_ml        = intel_sub_group_block_read2   (psl_u + 0u);
    const ushort4 sh_mh_d_dmin = intel_sub_group_block_read_us4(psl_s + 64u);
    const uint   sl = sl_ml.s0;
    const uint   ml = sl_ml.s1;
    const ushort sh = sh_mh_d_dmin.s0;
    const ushort mh = sh_mh_d_dmin.s1;
    const ushort dd = sh_mh_d_dmin.s2;
    const ushort dn = sh_mh_d_dmin.s3;
    const float d    = (float)as_half(dd);
    const float dmin = (float)as_half(dn);

    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 8; ++j) {
        const uint sq = ((sl >> (j * 4)) & 0xFu) | (((sh >> (j * 2)) & 0x3u) << 4);
        const uint mq = ((ml >> (j * 4)) & 0xFu) | (((mh >> (j * 2)) & 0x3u) << 4);
        const float scale = (float)sq * d, minv = (float)mq * dmin;
        const __private uchar* qh_row = qh_bytes + j * 4;   // 4 bytes: qh_b[j][0..3]
        __attribute__((opencl_unroll_hint))
        for (int kc = 0; kc < 4; ++kc) {
            __attribute__((opencl_unroll_hint))
            for (int kb = 0; kb < 4; ++kb) {
                const int b = kc * 4 + kb;
                const uchar byte = qs_bytes[j * 16 + b];
                const int k4 = b % 4, s4 = b / 4;
                const int bit_lo = (qh_row[k4] >> s4) & 1;
                const int bit_hi = (qh_row[k4] >> (4 + s4)) & 1;
                const int w0 = (int)(byte & 0x0F) + bit_lo * 16;
                const int w1 = (int)((byte >> 4) & 0x0F) + bit_hi * 16;
                out[j * 32 + b]      = (half)((float)w0 * scale - minv);
                out[j * 32 + 16 + b] = (half)((float)w1 * scale - minv);
            }
        }
    }
}
#endif

#if defined(GGUF_IS_Q6_K)
inline void FUNC(mtq_decode_block_sg)(const __global uchar* w_expert_base, int h, int lid, int bid,
                                      int blocks_per_row, __private half* out) {
    const uint num_blocks = (uint)N_SIZE * (uint)blocks_per_row;
    const __global uchar* ql_sec = w_expert_base;
    const __global uchar* qh_sec = ql_sec + (size_t)num_blocks * 128;
    const __global uchar* ps_sec = qh_sec + (size_t)num_blocks * 64;
    const __global uchar* pd_sec = ps_sec + (size_t)num_blocks * 16;

    const uint group_idx = (uint)h * (uint)blocks_per_row + (uint)bid;
    const __global uchar* ql_group = ql_sec + (size_t)group_idx * SG_OPG * 128;
    const __global uchar* qh_group = qh_sec + (size_t)group_idx * SG_OPG * 64;
    const __global uchar* ps_group = ps_sec + (size_t)group_idx * SG_OPG * 16;
    const __global uchar* pd_group = pd_sec + (size_t)group_idx * SG_OPG * 2;

    // ql: 128 B/lane = 32 uint chunks; qh: 64 B/lane = 16 uint chunks. One gather each.
    __private uint ql_u[32];
    FUNC_CALL(mtq_gather_plane)(ql_group, 0u, ql_u, 32);
    const __private uchar* ql_bytes = (const __private uchar*)ql_u;

    __private uint qh_u[16];
    FUNC_CALL(mtq_gather_plane)(qh_group, 0u, qh_u, 16);
    const __private uchar* qh_bytes = (const __private uchar*)qh_u;

    // ps: 16 scale bytes, scale si @ si*OPG + lid -> one uchar16 block read (component si -> lane lid).
    const uchar16 scq = intel_sub_group_block_read_uc16((const __global uchar*)ps_group);
    __private char sc[16];
    sc[0]  = as_char(scq.s0); sc[1]  = as_char(scq.s1); sc[2]  = as_char(scq.s2); sc[3]  = as_char(scq.s3);
    sc[4]  = as_char(scq.s4); sc[5]  = as_char(scq.s5); sc[6]  = as_char(scq.s6); sc[7]  = as_char(scq.s7);
    sc[8]  = as_char(scq.s8); sc[9]  = as_char(scq.s9); sc[10] = as_char(scq.sa); sc[11] = as_char(scq.sb);
    sc[12] = as_char(scq.sc); sc[13] = as_char(scq.sd); sc[14] = as_char(scq.se); sc[15] = as_char(scq.sf);

    // pd: one f16 per lane @ lid*2 -> one ushort block read.
    const ushort dbits = intel_sub_group_block_read_us((const __global ushort*)pd_group);
    const float d = (float)as_half(dbits);

    const int SH[16] = {0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30};

    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 8; ++j) {
        const __private uchar* qh_row  = qh_bytes + (2 * j + 0) * 4;   // hlo bytes
        const __private uchar* qh_row2 = qh_bytes + (2 * j + 1) * 4;   // hhi bytes
        const uint hlo = (uint)qh_row[0]  | ((uint)qh_row[1]  << 8) | ((uint)qh_row[2]  << 16) | ((uint)qh_row[3]  << 24);
        const uint hhi = (uint)qh_row2[0] | ((uint)qh_row2[1] << 8) | ((uint)qh_row2[2] << 16) | ((uint)qh_row2[3] << 24);
        const float clo = d * (float)sc[2 * j + 0];
        const float chi = d * (float)sc[2 * j + 1];

        __attribute__((opencl_unroll_hint))
        for (int kc = 0; kc < 4; ++kc) {
            __attribute__((opencl_unroll_hint))
            for (int kb = 0; kb < 4; ++kb) {
                const int l = kc * 4 + kb;
                const uchar byte = ql_bytes[j * 16 + l];
                const int w0 = (int)(byte & 0x0F) + (int)(((hlo >> SH[l]) & 3) * 16) - 32;
                const int w1 = (int)((byte >> 4) & 0x0F) + (int)(((hhi >> SH[l]) & 3) * 16) - 32;
                out[j * 32 + l]      = (half)((float)w0 * clo);
                out[j * 32 + 16 + l] = (half)((float)w1 * chi);
            }
        }
    }
}
#endif

#if defined(GGUF_IS_Q8_0)
// Decode ONE native 32-element Q8_0 block (block index `blk`) directly from the SG-packed
// [pqs | pd] shared-expert layout produced by RepackGGUFMoEWeights::pack_shared_q8_0_sg()
// (repack_gguf_moe_weights.cpp) -- the SAME byte layout shared_gate_up_q8_0 / shared_down_merge_q8_0
// (moe_gguf_sg_gemv.cl) decode at decode time. Q8_0 groups eight native 32-blocks into a 256-elem
// super-block; here `blk` is the native block index (blocks_per_row == K/32), so super-block
// super_bid = blk/8 and sub-block sub_j = blk%8. Numerically identical (w = q * d, signed, no min)
// to the raw-layout GGUF_IS_Q8_0 mtq_decode_block above; only the addressing/fetch strategy differs.
inline void FUNC(mtq_decode_block_sg)(const __global uchar* w_expert_base, int h, int lid, int blk,
                                      int blocks_per_row, __private half* out) {
    const uint nbpr      = (uint)blocks_per_row / 8u;      // 256-elem super-blocks per row
    const uint num_super = (uint)N_SIZE * nbpr;            // super-blocks in this projection
    const uint super_bid = (uint)blk / 8u;
    const uint sub_j     = (uint)blk % 8u;

    const __global uchar* pqs_sec = w_expert_base;
    const __global uchar* pd_sec  = pqs_sec + (size_t)num_super * 256;

    const uint group_idx = (uint)h * nbpr + super_bid;
    const __global uchar* pqs_group = pqs_sec + (size_t)group_idx * SG_OPG * 256;
    const __global uchar* pd_group  = pd_sec  + (size_t)group_idx * SG_OPG * 16;

    // pqs: 256 B/lane = 64 uint chunks -> one gather (8 wide block_read8 transactions).
    __private uint pqs_u[64];
    FUNC_CALL(mtq_gather_plane)(pqs_group, 0u, pqs_u, 64);
    const __private char* pqs_bytes = (const __private char*)pqs_u;

    // Sub-block scale: SoA field sub_j at sub_j*OPG*2 + lid*2 (f16) -> one ushort block read.
    const __global ushort* pd_s = (const __global ushort*)pd_group;
    const ushort dbits = intel_sub_group_block_read_us(pd_s + sub_j * (uint)SG_OPG);
    const float d = (float)as_half(dbits);

    // 8 chunks x 4 signed int8 = 32 weights, gathered above at chunk index (sub_j*8 + c).
    __attribute__((opencl_unroll_hint))
    for (int c = 0; c < 8; ++c) {
        __attribute__((opencl_unroll_hint))
        for (int m = 0; m < 4; ++m)
            out[c * 4 + m] = (half)((float)pqs_bytes[(sub_j * 8u + (uint)c) * 4 + (uint)m] * d);
    }
}
#endif

#endif  // GGUF_SG_LAYOUT

// ---- main transcode kernel ----
__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(moe_gguf_transcode)(
    const __global uchar* W,
          __global uchar* WQ,
          __global half*  SC
)
{
    const int blocks_per_row = K_SIZE / GGUF_BLOCK_ELEM;
    const int n   = (int)get_global_id(0);
    const int blk = (int)get_global_id(1);
    const int e   = (int)get_global_id(2);
    if (n >= N_SIZE || blk >= blocks_per_row || e >= NUM_EXPERTS)
        return;

    // ---- Stage 1: decode ONE GGUF block into private registers. ----
    half blk_vals[GGUF_BLOCK_ELEM];
#if GGUF_SG_LAYOUT
    // SG-packed layout: this work-item's row n = h*SG_OPG + lid, with h == get_group_id(0) and
    // lid == get_local_id(0) directly (dispatch local size == SG_OPG == 16, see header comment).
    const int h_grp = (int)get_group_id(0);
    const int lid   = (int)get_local_id(0);
    const __global uchar* w_expert_base = W + (size_t)e * (size_t)N_SIZE * (size_t)blocks_per_row * GGUF_BLOCK_BYTES;
    FUNC_CALL(mtq_decode_block_sg)(w_expert_base, h_grp, lid, blk, blocks_per_row, blk_vals);
#else
    const __global uchar* w_row =
        W + ((uint)e * (uint)N_SIZE + (uint)n) * (uint)blocks_per_row * GGUF_BLOCK_BYTES;
    FUNC_CALL(mtq_decode_block)(w_row + (uint)blk * GGUF_BLOCK_BYTES, blk_vals);
#endif

    const uint row_base       = ((uint)e * (uint)N_SIZE + (uint)n) * (uint)K_SIZE;
    const uint groups_per_row = (uint)K_SIZE / (uint)REQUANT_GROUP;
    const uint sc_expert_base = (uint)e * groups_per_row * (uint)N_SIZE;

    const int groups_per_block = GGUF_BLOCK_ELEM / REQUANT_GROUP;
#if !TRANSCODE_TO_I4
    __global char* wq_i8 = (__global char*)WQ;
#endif

    // ---- Stage 2: fused amax + vectorized requantize per group. ----
    __attribute__((opencl_unroll_hint))
    for (int gi = 0; gi < groups_per_block; ++gi) {
        const int off_in_blk = gi * REQUANT_GROUP;
        const int g          = blk * groups_per_block + gi;
        const int k0         = g * REQUANT_GROUP;

        // Materialize the group into a private float window ONCE (kills scratch reload
        // of blk_vals — the SBID stall source). 4-way fmax breaks the reduction chain.
        float gvf[REQUANT_GROUP];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 4) {
            const float v0 = (float)blk_vals[off_in_blk + i + 0];
            const float v1 = (float)blk_vals[off_in_blk + i + 1];
            const float v2 = (float)blk_vals[off_in_blk + i + 2];
            const float v3 = (float)blk_vals[off_in_blk + i + 3];
            gvf[i + 0] = v0; gvf[i + 1] = v1; gvf[i + 2] = v2; gvf[i + 3] = v3;
            a0 = fmax(a0, fabs(v0));
            a1 = fmax(a1, fabs(v1));
            a2 = fmax(a2, fabs(v2));
            a3 = fmax(a3, fabs(v3));
        }
        const float amax = fmax(fmax(a0, a1), fmax(a2, a3));

        const float scale     = (amax > 0.0f) ? (amax * (1.0f / (float)QMAX)) : 1.0f;
        const float inv_scale = (amax > 0.0f) ? ((float)QMAX * native_recip(amax)) : 0.0f;

        // Coalesced scale store across the sub-group (adjacent n -> adjacent addr).
        SC[sc_expert_base + (uint)g * (uint)N_SIZE + (uint)n] = (half)scale;

        // Pre-materialize quantized ints. VECTORIZED over float8/int8 blocks so the
        // compiler emits packed SIMD multiply + round + clamp — fewer ALU1 instructions
        // and a shorter dependency chain, directly attacking the SBID/Dist-Acc stalls.
        int q[REQUANT_GROUP];
#if TRANSCODE_TO_I4
    #if (REQUANT_GROUP % 8) == 0
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 8) {
            float8 fv = (float8)(gvf[i+0], gvf[i+1], gvf[i+2], gvf[i+3],
                                 gvf[i+4], gvf[i+5], gvf[i+6], gvf[i+7]);
            int8 qi = convert_int8(round(fv * inv_scale));
            qi = clamp(qi, (int8)(-8), (int8)(7)) & (int8)(0x0F);
            q[i+0]=qi.s0; q[i+1]=qi.s1; q[i+2]=qi.s2; q[i+3]=qi.s3;
            q[i+4]=qi.s4; q[i+5]=qi.s5; q[i+6]=qi.s6; q[i+7]=qi.s7;
        }
    #else
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; ++i)
            q[i] = clamp((int)round(gvf[i] * inv_scale), -8, 7) & 0x0F;
    #endif

        const uint byte_base = (row_base + (uint)k0) >> 1;
        int i = 0;
    #if (REQUANT_GROUP % 8) == 0
        __global uchar4* wq4 = (__global uchar4*)(WQ + byte_base);
        __attribute__((opencl_unroll_hint))
        for (; i <= REQUANT_GROUP - 8; i += 8) {
            wq4[i >> 3] = (uchar4)(
                (uchar)(q[i + 0] | (q[i + 1] << 4)),
                (uchar)(q[i + 2] | (q[i + 3] << 4)),
                (uchar)(q[i + 4] | (q[i + 5] << 4)),
                (uchar)(q[i + 6] | (q[i + 7] << 4)));
        }
    #endif
        __attribute__((opencl_unroll_hint))
        for (; i < REQUANT_GROUP; i += 2)
            WQ[byte_base + ((uint)i >> 1)] = (uchar)(q[i] | (q[i + 1] << 4));
#else
    #if (REQUANT_GROUP % 8) == 0
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; i += 8) {
            float8 fv = (float8)(gvf[i+0], gvf[i+1], gvf[i+2], gvf[i+3],
                                 gvf[i+4], gvf[i+5], gvf[i+6], gvf[i+7]);
            int8 qi = convert_int8(round(fv * inv_scale));
            qi = clamp(qi, (int8)(-128), (int8)(127));
            q[i+0]=qi.s0; q[i+1]=qi.s1; q[i+2]=qi.s2; q[i+3]=qi.s3;
            q[i+4]=qi.s4; q[i+5]=qi.s5; q[i+6]=qi.s6; q[i+7]=qi.s7;
        }
    #else
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < REQUANT_GROUP; ++i)
            q[i] = clamp((int)round(gvf[i] * inv_scale), -128, 127);
    #endif

        const uint wq_base = row_base + (uint)k0;
        int i = 0;
    #if (REQUANT_GROUP % 4) == 0
        __global char4* wq4 = (__global char4*)(wq_i8 + wq_base);
        __attribute__((opencl_unroll_hint))
        for (; i <= REQUANT_GROUP - 4; i += 4)
            wq4[i >> 2] = (char4)((char)q[i], (char)q[i + 1], (char)q[i + 2], (char)q[i + 3]);
    #endif
        __attribute__((opencl_unroll_hint))
        for (; i < REQUANT_GROUP; ++i)
            wq_i8[wq_base + (uint)i] = (char)q[i];
#endif
    }
}
