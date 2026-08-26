// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// moe_gguf_sg_gemv.cl -- High-performance GGUF MoE GEMV kernels using the transposed
// "SG" (sub-group block-read) weight layout, for Q4_K / Q5_K / Q6_K.
//
// This file replaces, for these three block-quant types only, the batched-GEMV decode
// path previously served by moe_3gemm_swiglu_mlp.cl (WEIGHT_COMPRESSEION_DT == 3,
// GATEUP_DECODE/DOWN_DECODE in {1,2,3}). Q4_0 / Q8_0 (decode code 4) and every other
// weight compression type are unaffected and continue to use moe_3gemm_swiglu_mlp.cl.
//
// Ported from q4k_moe_gemv/moe_gemv_q{4,5,6}k_sg.cl (reference host harness:
// q4k_moe_gemv/test_moe_gemv_sg_kernels.py). Numerics, dispatch grid (OPG=16 lanes,
// KSPLIT reduction sub-groups) and the transposed per-expert weight byte layout are
// unchanged from the reference; only the entry points, argument list and routing/
// merge integration were adapted to this project's MoE op (see
// ocl_v2/moe/moe_3gemm_swiglu_opt.cpp exec_batched_gemv()).
//
// ============================================================================
// Weight layout (per expert, "SG" packed -- see RepackGGUFMoEWeights in
// repack_gguf_moe_weights.cpp for the host-side byte transform applied once at graph
// compile time; identical byte layout to pack_q{4,5,6}k_sg() in the python reference):
//
//   For an expert weight matrix [N, K] with nbpr = K/256 blocks-per-row and
//   nrg = N/OPG row-groups (OPG = 16):
//
//   Q4_K expert_size = nbpr*N*144 bytes = pqs_T (nbpr*N*128) + psl_T (nbpr*N*16)
//   Q5_K expert_size = nbpr*N*176 bytes = pqs_T (128) + pqh_T (32) + psl_T (16), all *nbpr*N
//   Q6_K expert_size = nbpr*N*210 bytes = pql_T (128) + pqh_T (64) + ps_T (16) + pd_T (2), all *nbpr*N
//
//   All sections are addressed as (h*nbpr + bid) * OPG * <section_bytes> where
//   h = row-group index (row_group = n / OPG), bid = block-in-row index.
//
// ============================================================================
// Routing / merge integration:
//   gate_up_sg  : one work-group per (token, topk-slot); output written to the same
//                 flattened [token_num * EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE] scratch
//                 buffer (scratch.up) that the existing architecture's mlp_down stage
//                 consumes -- MERGE_KSPLIT reduction happens inside the work-group.
//   down_merge_sg: FUSES the existing mlp_down + mlp_reduce stages: one work-group per
//                 (token, output-row-group) loops over ALL EXPERTS_PER_TOKEN experts of
//                 that token internally (weighted by the routing weight / "efficient"),
//                 and writes the final per-token hidden_states row directly. No separate
//                 reduce stage or intermediate scratch.y is used for this path.
//
// Current scope / limitations (opt-in path, see moe_3gemm_swiglu_opt.cpp
// gguf_moe_sg_enabled()):
//   - Shared-expert MoE (num_shared_expert > 0) IS handled, but only when the shared
//     expert's gate/up/down weights are GGUF Q8_0 (see the "Shared-expert Q8_0 kernels"
//     section at the end of this file): shared_gate_scalar_q8_0 / shared_gate_up_q8_0 /
//     shared_down_merge_q8_0. Any other shared-expert weight type falls back entirely to
//     moe_3gemm_swiglu_mlp.cl (routed experts included), same as before this support was
//     added -- see moe_3gemm_swiglu_opt.cpp's shared_expert_sg_ok check.
//   - Activations / gate_up intermediate buffer / final output are assumed fp16 (the
//     universal case for GGUF-quantized models in this codebase); MOE_DTYPE == float is
//     not supported by this path and the caller must not select it in that case.
//   - The "fast hoisted routing" micro-opt from the reference (single sub-group block
//     read of the whole per-token routing table) is intentionally NOT ported: MAX_TOPK is
//     always << OPG(16) in practice, so that micro-opt never triggers; dropping it removes
//     a class of layout bugs (it required the routing-weight buffer to be 4-byte aligned
//     f32, but this project's routing weight scratch is fp16).
// ============================================================================

#pragma OPENCL EXTENSION cl_khr_fp16        : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

// `unroll_for` (and the other shared jitter macros) live in the batch header
// cl_kernels/include/batch_headers/common.cl. Batch headers are NOT prepended to a
// program automatically -- ocl_v2's kernels_db_gen.py only attaches a batch header to
// this file's program if the file actually `#include`s it (process_includes() hoists
// referenced `batch_headers/*` includes to the top of the stored source string, and the
// runtime resolves only those). Without this include, every program built from this file
// (gate_up_sg / down_merge_sg / shared_*_q8_0) fails with
//   "implicit declaration of function 'unroll_for'".
// Including it here (exactly like moe_3gemm_swiglu_mlp.cl) is safe: batch-header bodies are
// never inlined into this file's text, so add_missing_undefs() cannot emit a stray
// `#undef unroll_for`; it also transitively pulls common.cl (unroll_for) via its own
// `#include "common.cl"`.
#include "include/batch_headers/sub_group_block_read.cl"

#define OPG 16 /* = sub_group_size, fixed by the reference kernels */

#ifndef KSPLIT
#define KSPLIT 1
#endif

#ifndef SPLIT_UG
#if KSPLIT >= 2 && (KSPLIT % 2) == 0
#define SPLIT_UG 1
#else
#define SPLIT_UG 0
#endif
#endif

// NOTE: `unroll_for` is provided by the batch header included above
// (include/batch_headers/common.cl, pulled in via sub_group_block_read.cl). Do NOT
// `#define unroll_for` in this file: ocl_v2's kernels_db_gen.py appends an `#undef` for
// every macro `#define`d in the file body, which would strip the definition again for the
// next batched KERNEL copy. Relying on the batch-header include (whose body is never inlined
// into this file) avoids that.

#ifndef DM_BLOCKED
#define DM_BLOCKED 1
#endif

#define MOE_SG_Q4_K 1
#define MOE_SG_Q5_K 2
#define MOE_SG_Q6_K 3

// ============================================================================
// Stage selector: exactly one of GATE_UP_SG_ENABLE / DOWN_MERGE_SG_ENABLE is set to 1
// by the host JIT per compiled program (mirrors the GATE_UP_ENABLE/DOWN_ENABLE pattern
// in moe_3gemm_swiglu_mlp.cl). This guarantees only one KERNEL(...) definition survives
// preprocessing (KERNEL(name) always expands to "__kernel void <entry_point>" for
// whichever stage is being compiled, so two textual KERNEL(...) invocations in the same
// translation unit would otherwise collide on the same generated name).
// ============================================================================

#if GATE_UP_SG_ENABLE
#define QCODE GATEUP_DECODE
#elif DOWN_MERGE_SG_ENABLE
#define QCODE DOWN_DECODE
#endif

// ============================================================================================
// ============================  Q4_K  =========================================================
// ============================================================================================
#if defined(QCODE) && QCODE == MOE_SG_Q4_K

#define W2_MASK 0x000F000Fu

inline float FUNC(sg_q4k_subblock)(uint q0, uint q1, uint q2, uint q3, __local const float *in) {
    float4 vlo = (float4)(0.0f), vhi = (float4)(0.0f);
#define SG_Q4K_CHUNK(q, c)                                                                       \
    do {                                                                                          \
        uint _q = (q);                                                                            \
        uint _a =  _q        & W2_MASK;                                                           \
        uint _b = (_q >>  8) & W2_MASK;                                                           \
        uint _c = (_q >>  4) & W2_MASK;                                                           \
        uint _d = (_q >> 12) & W2_MASK;                                                           \
        float4 _wl = (float4)((float)as_ushort2(_a).x, (float)as_ushort2(_b).x,                   \
                               (float)(_a >> 16),       (float)(_b >> 16));                        \
        float4 _wh = (float4)((float)as_ushort2(_c).x, (float)as_ushort2(_d).x,                   \
                               (float)(_c >> 16),       (float)(_d >> 16));                        \
        vlo = fma(*(__local const float4 *)(in + (c) * 4),      _wl, vlo);                         \
        vhi = fma(*(__local const float4 *)(in + 16 + (c) * 4), _wh, vhi);                         \
    } while (0)
    SG_Q4K_CHUNK(q0, 0); SG_Q4K_CHUNK(q1, 1); SG_Q4K_CHUNK(q2, 2); SG_Q4K_CHUNK(q3, 3);
#undef SG_Q4K_CHUNK
    float4 s = vlo + vhi;
    return (s.x + s.y) + (s.z + s.w);
}

inline float FUNC(sg_q4k_acc)(uint8 w0, uint8 w1, uint8 w2, uint8 w3,
                        const float *scale, const float *minv, const float *isum,
                        __local const float *sb) {
    float acc = 0.0f;
    acc = fma(scale[0], FUNC_CALL(sg_q4k_subblock)(w0.s0, w0.s1, w0.s2, w0.s3, sb + 0 * 32), acc);
    acc = fma(scale[1], FUNC_CALL(sg_q4k_subblock)(w0.s4, w0.s5, w0.s6, w0.s7, sb + 1 * 32), acc);
    acc = fma(scale[2], FUNC_CALL(sg_q4k_subblock)(w1.s0, w1.s1, w1.s2, w1.s3, sb + 2 * 32), acc);
    acc = fma(scale[3], FUNC_CALL(sg_q4k_subblock)(w1.s4, w1.s5, w1.s6, w1.s7, sb + 3 * 32), acc);
    acc = fma(scale[4], FUNC_CALL(sg_q4k_subblock)(w2.s0, w2.s1, w2.s2, w2.s3, sb + 4 * 32), acc);
    acc = fma(scale[5], FUNC_CALL(sg_q4k_subblock)(w2.s4, w2.s5, w2.s6, w2.s7, sb + 5 * 32), acc);
    acc = fma(scale[6], FUNC_CALL(sg_q4k_subblock)(w3.s0, w3.s1, w3.s2, w3.s3, sb + 6 * 32), acc);
    acc = fma(scale[7], FUNC_CALL(sg_q4k_subblock)(w3.s4, w3.s5, w3.s6, w3.s7, sb + 7 * 32), acc);
    unroll_for (int j = 0; j < 8; j++) acc = fma(-minv[j], isum[j], acc);
    return acc;
}

inline void FUNC(sg_q4k_decode)(const __global uint *psl_pu, uint off, uint hh, float *scale, float *minv) {
    uint4 r = intel_sub_group_block_read4(psl_pu + off);
    const uint half_sel = (hh & 1u) * 16u;
    const uint src_lane = hh >> 1;
    uint shmh_lo = intel_sub_group_shuffle(r.s2, src_lane);
    uint shmh_hi = intel_sub_group_shuffle(r.s2, src_lane + 8u);
    uint ddmn_lo = intel_sub_group_shuffle(r.s3, src_lane);
    uint ddmn_hi = intel_sub_group_shuffle(r.s3, src_lane + 8u);
    uint sl = r.s0;
    uint ml = r.s1;
    uint sh = (shmh_lo >> half_sel) & 0xFFFFu;
    uint mh = (shmh_hi >> half_sel) & 0xFFFFu;
    float d    = (float)as_half((ushort)((ddmn_lo >> half_sel) & 0xFFFFu));
    float dmin = (float)as_half((ushort)((ddmn_hi >> half_sel) & 0xFFFFu));
    unroll_for (int i = 0; i < 8; i++) {
        uint sq = ((sl >> (i * 4)) & 0xFu) | (((sh >> (i * 2)) & 0x3u) << 4);
        uint mq = ((ml >> (i * 4)) & 0xFu) | (((mh >> (i * 2)) & 0x3u) << 4);
        scale[i] = (float)sq * d;
        minv[i]  = (float)mq * dmin;
    }
}

inline void FUNC(sg_q4k_load_inp)(__local float *sb, const __global half *src, uint hh, float *isum) {
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define SG_Q4K_STAGE(j, comp)                                                                     \
    {                                                                                             \
        float2 _f = convert_float2(as_half2(comp));                                               \
        *(__local float2 *)(sb + (j) * 32 + 2 * hh) = _f;                                         \
        isum[j] = sub_group_reduce_add(_f.x + _f.y);                                              \
    }
    SG_Q4K_STAGE(0, iv.s0) SG_Q4K_STAGE(1, iv.s1) SG_Q4K_STAGE(2, iv.s2) SG_Q4K_STAGE(3, iv.s3)
    SG_Q4K_STAGE(4, iv.s4) SG_Q4K_STAGE(5, iv.s5) SG_Q4K_STAGE(6, iv.s6) SG_Q4K_STAGE(7, iv.s7)
#undef SG_Q4K_STAGE
}

#define SG_EXPERT_BYTES(nb) ((nb) * 144)

#endif  // QCODE == MOE_SG_Q4_K

// ============================================================================================
// ============================  Q5_K  =========================================================
// ============================================================================================
#if defined(QCODE) && QCODE == MOE_SG_Q5_K

#define W2_MASK 0x000F000Fu
#define QH_MASK 0x00100010u

inline float FUNC(sg_q5k_subblock)(uint q0, uint q1, uint q2, uint q3, uint H, __local const float *in) {
    float4 vlo = (float4)(0.0f), vhi = (float4)(0.0f);
#define SG_Q5K_CHUNK(q, c)                                                                        \
    do {                                                                                          \
        uint _q = (q);                                                                            \
        uint _a = ( _q        & W2_MASK) | ((H << (4 - (c))) & QH_MASK);                          \
        uint _b = ((_q >>  8) & W2_MASK) | ((H >> (4 + (c))) & QH_MASK);                          \
        uint _c = ((_q >>  4) & W2_MASK) | ((H >>      (c) ) & QH_MASK);                          \
        uint _d = ((_q >> 12) & W2_MASK) | ((H >> (8 + (c))) & QH_MASK);                          \
        float4 _wl = (float4)((float)as_ushort2(_a).x, (float)as_ushort2(_b).x,                   \
                               (float)(_a >> 16),       (float)(_b >> 16));                        \
        float4 _wh = (float4)((float)as_ushort2(_c).x, (float)as_ushort2(_d).x,                   \
                               (float)(_c >> 16),       (float)(_d >> 16));                        \
        vlo = fma(*(__local const float4 *)(in + (c) * 4),      _wl, vlo);                         \
        vhi = fma(*(__local const float4 *)(in + 16 + (c) * 4), _wh, vhi);                         \
    } while (0)
    SG_Q5K_CHUNK(q0, 0); SG_Q5K_CHUNK(q1, 1); SG_Q5K_CHUNK(q2, 2); SG_Q5K_CHUNK(q3, 3);
#undef SG_Q5K_CHUNK
    float4 s = vlo + vhi;
    return (s.x + s.y) + (s.z + s.w);
}

inline float FUNC(sg_q5k_acc)(uint8 w0, uint8 w1, uint8 w2, uint8 w3, uint8 hq,
                        const float *scale, const float *minv, const float *isum,
                        __local const float *sb) {
    float acc = 0.0f;
    acc = fma(scale[0], FUNC_CALL(sg_q5k_subblock)(w0.s0, w0.s1, w0.s2, w0.s3, hq.s0, sb + 0 * 32), acc);
    acc = fma(scale[1], FUNC_CALL(sg_q5k_subblock)(w0.s4, w0.s5, w0.s6, w0.s7, hq.s1, sb + 1 * 32), acc);
    acc = fma(scale[2], FUNC_CALL(sg_q5k_subblock)(w1.s0, w1.s1, w1.s2, w1.s3, hq.s2, sb + 2 * 32), acc);
    acc = fma(scale[3], FUNC_CALL(sg_q5k_subblock)(w1.s4, w1.s5, w1.s6, w1.s7, hq.s3, sb + 3 * 32), acc);
    acc = fma(scale[4], FUNC_CALL(sg_q5k_subblock)(w2.s0, w2.s1, w2.s2, w2.s3, hq.s4, sb + 4 * 32), acc);
    acc = fma(scale[5], FUNC_CALL(sg_q5k_subblock)(w2.s4, w2.s5, w2.s6, w2.s7, hq.s5, sb + 5 * 32), acc);
    acc = fma(scale[6], FUNC_CALL(sg_q5k_subblock)(w3.s0, w3.s1, w3.s2, w3.s3, hq.s6, sb + 6 * 32), acc);
    acc = fma(scale[7], FUNC_CALL(sg_q5k_subblock)(w3.s4, w3.s5, w3.s6, w3.s7, hq.s7, sb + 7 * 32), acc);
    unroll_for (int j = 0; j < 8; j++) acc = fma(-minv[j], isum[j], acc);
    return acc;
}

inline void FUNC(sg_q5k_decode)(const __global uint *psl_pu, uint off, uint hh, float *scale, float *minv) {
    uint4 r = intel_sub_group_block_read4(psl_pu + off);
    const uint half_sel = (hh & 1u) * 16u;
    const uint src_lane = hh >> 1;
    uint shmh_lo = intel_sub_group_shuffle(r.s2, src_lane);
    uint shmh_hi = intel_sub_group_shuffle(r.s2, src_lane + 8u);
    uint ddmn_lo = intel_sub_group_shuffle(r.s3, src_lane);
    uint ddmn_hi = intel_sub_group_shuffle(r.s3, src_lane + 8u);
    uint sl = r.s0;
    uint ml = r.s1;
    uint sh = (shmh_lo >> half_sel) & 0xFFFFu;
    uint mh = (shmh_hi >> half_sel) & 0xFFFFu;
    float d    = (float)as_half((ushort)((ddmn_lo >> half_sel) & 0xFFFFu));
    float dmin = (float)as_half((ushort)((ddmn_hi >> half_sel) & 0xFFFFu));
    unroll_for (int i = 0; i < 8; i++) {
        uint sq = ((sl >> (i * 4)) & 0xFu) | (((sh >> (i * 2)) & 0x3u) << 4);
        uint mq = ((ml >> (i * 4)) & 0xFu) | (((mh >> (i * 2)) & 0x3u) << 4);
        scale[i] = (float)sq * d;
        minv[i]  = (float)mq * dmin;
    }
}

inline void FUNC(sg_q5k_load_inp)(__local float *sb, const __global half *src, uint hh, float *isum) {
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define SG_Q5K_STAGE(j, comp)                                                                     \
    {                                                                                             \
        float2 _f = convert_float2(as_half2(comp));                                               \
        *(__local float2 *)(sb + (j) * 32 + 2 * hh) = _f;                                         \
        isum[j] = sub_group_reduce_add(_f.x + _f.y);                                              \
    }
    SG_Q5K_STAGE(0, iv.s0) SG_Q5K_STAGE(1, iv.s1) SG_Q5K_STAGE(2, iv.s2) SG_Q5K_STAGE(3, iv.s3)
    SG_Q5K_STAGE(4, iv.s4) SG_Q5K_STAGE(5, iv.s5) SG_Q5K_STAGE(6, iv.s6) SG_Q5K_STAGE(7, iv.s7)
#undef SG_Q5K_STAGE
}

#define SG_EXPERT_BYTES(nb) ((nb) * 176)

#endif  // QCODE == MOE_SG_Q5_K

// ============================================================================================
// ============================  Q6_K  =========================================================
// ============================================================================================
#if defined(QCODE) && QCODE == MOE_SG_Q6_K

#define W2_MASK 0x000F000Fu
#define Q6_MASK 0x00300030u

inline float2 FUNC(sg_q6k_subblock)(uint q0, uint q1, uint q2, uint q3, uint HL, uint HH, __local const float *in) {
    float4 vlo = (float4)(0.0f), vhi = (float4)(0.0f);
#define SG_Q6K_CHUNK(q, c)                                                                        \
    do {                                                                                          \
        uint _q = (q);                                                                            \
        uint _a = ( _q        & W2_MASK) | (((HL) << 4) >> (2 * (c)) & Q6_MASK);                  \
        uint _b = ((_q >>  8) & W2_MASK) | ((HL) >> (4 + 2 * (c))    & Q6_MASK);                  \
        uint _c = ((_q >>  4) & W2_MASK) | (((HH) << 4) >> (2 * (c)) & Q6_MASK);                  \
        uint _d = ((_q >> 12) & W2_MASK) | ((HH) >> (4 + 2 * (c))    & Q6_MASK);                  \
        float4 _wl = (float4)((float)as_ushort2(_a).x, (float)as_ushort2(_b).x,                   \
                               (float)(_a >> 16),       (float)(_b >> 16));                        \
        float4 _wh = (float4)((float)as_ushort2(_c).x, (float)as_ushort2(_d).x,                   \
                               (float)(_c >> 16),       (float)(_d >> 16));                        \
        vlo = fma(*(__local const float4 *)(in + (c) * 4),      _wl, vlo);                         \
        vhi = fma(*(__local const float4 *)(in + 16 + (c) * 4), _wh, vhi);                         \
    } while (0)
    SG_Q6K_CHUNK(q0, 0); SG_Q6K_CHUNK(q1, 1); SG_Q6K_CHUNK(q2, 2); SG_Q6K_CHUNK(q3, 3);
#undef SG_Q6K_CHUNK
    return (float2)((vlo.x + vlo.y) + (vlo.z + vlo.w), (vhi.x + vhi.y) + (vhi.z + vhi.w));
}

#define SG_Q6K_SUB(j, a, b, c, d, hl, hh)                                                          \
    do {                                                                                          \
        float2 _dv = FUNC_CALL(sg_q6k_subblock)(a, b, c, d, hl, hh, sb + (j) * 32);                          \
        acc = fma(clo[j], _dv.x, acc);                                                            \
        acc = fma(chi[j], _dv.y, acc);                                                            \
    } while (0)

inline float FUNC(sg_q6k_acc)(uint8 w0, uint8 w1, uint8 w2, uint8 w3, uint8 hq0, uint8 hq1,
                        const float *clo, const float *chi, __local const float *sb) {
    float acc = 0.0f;
    SG_Q6K_SUB(0, w0.s0, w0.s1, w0.s2, w0.s3, hq0.s0, hq0.s1);
    SG_Q6K_SUB(1, w0.s4, w0.s5, w0.s6, w0.s7, hq0.s2, hq0.s3);
    SG_Q6K_SUB(2, w1.s0, w1.s1, w1.s2, w1.s3, hq0.s4, hq0.s5);
    SG_Q6K_SUB(3, w1.s4, w1.s5, w1.s6, w1.s7, hq0.s6, hq0.s7);
    SG_Q6K_SUB(4, w2.s0, w2.s1, w2.s2, w2.s3, hq1.s0, hq1.s1);
    SG_Q6K_SUB(5, w2.s4, w2.s5, w2.s6, w2.s7, hq1.s2, hq1.s3);
    SG_Q6K_SUB(6, w3.s0, w3.s1, w3.s2, w3.s3, hq1.s4, hq1.s5);
    SG_Q6K_SUB(7, w3.s4, w3.s5, w3.s6, w3.s7, hq1.s6, hq1.s7);
    return acc;
}

inline void FUNC(sg_q6k_decode)(uchar16 scq, ushort d_raw, float *clo, float *chi) {
    float d = (float)as_half(d_raw);
    char sc[16];
    sc[ 0]=as_char(scq.s0); sc[ 1]=as_char(scq.s1); sc[ 2]=as_char(scq.s2); sc[ 3]=as_char(scq.s3);
    sc[ 4]=as_char(scq.s4); sc[ 5]=as_char(scq.s5); sc[ 6]=as_char(scq.s6); sc[ 7]=as_char(scq.s7);
    sc[ 8]=as_char(scq.s8); sc[ 9]=as_char(scq.s9); sc[10]=as_char(scq.sa); sc[11]=as_char(scq.sb);
    sc[12]=as_char(scq.sc); sc[13]=as_char(scq.sd); sc[14]=as_char(scq.se); sc[15]=as_char(scq.sf);
    unroll_for (int j = 0; j < 8; j++) {
        clo[j] = d * (float)sc[2 * j];
        chi[j] = d * (float)sc[2 * j + 1];
    }
}

inline uint8 FUNC(sg_q6k_stage_inp)(__local float *sb, const __global half *src, uint hh) {
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define SG_Q6K_ST(j, comp) *(__local float2 *)(sb + (j) * 32 + 2 * hh) = convert_float2(as_half2(comp));
    SG_Q6K_ST(0, iv.s0) SG_Q6K_ST(1, iv.s1) SG_Q6K_ST(2, iv.s2) SG_Q6K_ST(3, iv.s3)
    SG_Q6K_ST(4, iv.s4) SG_Q6K_ST(5, iv.s5) SG_Q6K_ST(6, iv.s6) SG_Q6K_ST(7, iv.s7)
#undef SG_Q6K_ST
    return iv;
}

inline float FUNC(sg_q6k_corr)(uint8 iv, const float *clo, const float *chi) {
    float corr = 0.0f;
#define SG_Q6K_STAGE(j, comp)                                                                      \
    {                                                                                             \
        float2 _f = convert_float2(as_half2(comp));                                               \
        float _v = _f.x + _f.y;                                                                   \
        _v += intel_sub_group_shuffle_xor(_v, 1);                                                  \
        _v += intel_sub_group_shuffle_xor(_v, 2);                                                  \
        _v += intel_sub_group_shuffle_xor(_v, 4);                                                  \
        corr = fma(clo[j], intel_sub_group_shuffle(_v, 0), corr);                                  \
        corr = fma(chi[j], intel_sub_group_shuffle(_v, 8), corr);                                  \
    }
    SG_Q6K_STAGE(0, iv.s0) SG_Q6K_STAGE(1, iv.s1) SG_Q6K_STAGE(2, iv.s2) SG_Q6K_STAGE(3, iv.s3)
    SG_Q6K_STAGE(4, iv.s4) SG_Q6K_STAGE(5, iv.s5) SG_Q6K_STAGE(6, iv.s6) SG_Q6K_STAGE(7, iv.s7)
#undef SG_Q6K_STAGE
    return corr;
}

#define SG_EXPERT_BYTES(nb) ((nb) * 210)

#endif  // QCODE == MOE_SG_Q6_K

// ============================================================================================
// ==========================  Kernel 1: gate_up_sg  ===========================================
// One work-group per (token t, topk-slot v). Writes scratch.up[(t*EXPERTS_PER_TOKEN+v)*
// INTERMEDIATE_SIZE + row] = silu(gate) * up, same flatten convention consumed by
// down_merge_sg (and, for other weight types, by the existing mlp_down stage).
// global = (INTERMEDIATE_SIZE, KSPLIT, EXPERTS_PER_TOKEN * token_num)
// local  = (OPG=16, KSPLIT, 1)
// ============================================================================================
#if GATE_UP_SG_ENABLE

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(gate_up_sg)(
    const __global uint  * restrict expert_list,  /* [token_num, EXPERTS_PER_TOKEN] */
    const __global uchar * restrict gate_weights,  /* [EXPERT_NUM * expert_size] */
    const __global uchar * restrict up_weights,    /* [EXPERT_NUM * expert_size] */
    const __global half  * restrict hidden_states, /* [token_num, HIDDEN_SIZE] */
          __global half  * restrict gate_up_out    /* [token_num*EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE] */
) {
    const uint output_num  = EXPERTS_PER_TOKEN;
    const uint input_len   = HIDDEN_SIZE;
    const uint output_len  = INTERMEDIATE_SIZE;
    const uint nbpr        = input_len / 256;
    const uint num_blocks  = input_len * output_len / 256;
    const uint expert_size = SG_EXPERT_BYTES(num_blocks);

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint vt = get_group_id(2);
    const uint v  = vt % output_num;
    const uint t  = vt / output_num;

    const uint index = expert_list[t * output_num + v];

    const __global uchar *up_pqs = up_weights   + (size_t)index * expert_size;
    const __global uchar *gt_pqs = gate_weights + (size_t)index * expert_size;

    __local float slm_inp [KSPLIT * 2 * 256];
    __local float slm_up  [KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

#if QCODE == MOE_SG_Q4_K
    const __global uint *up_pqs_u  = (const __global uint *)up_pqs;
    const __global uint *up_psl_pu = (const __global uint *)(up_pqs + (size_t)num_blocks * 128);
    const __global uint *gt_pqs_u  = (const __global uint *)gt_pqs;
    const __global uint *gt_psl_pu = (const __global uint *)(gt_pqs + (size_t)num_blocks * 128);
#elif QCODE == MOE_SG_Q5_K
    const __global uint *up_pqs_u  = (const __global uint *)up_pqs;
    const __global uint *up_pqh_u  = (const __global uint *)(up_pqs + (size_t)num_blocks * 128);
    const __global uint *up_psl_pu = (const __global uint *)(up_pqs + (size_t)num_blocks * 160);
    const __global uint *gt_pqs_u  = (const __global uint *)gt_pqs;
    const __global uint *gt_pqh_u  = (const __global uint *)(gt_pqs + (size_t)num_blocks * 128);
    const __global uint *gt_psl_pu = (const __global uint *)(gt_pqs + (size_t)num_blocks * 160);
#elif QCODE == MOE_SG_Q6_K
    const __global uint  *up_pql_u = (const __global uint  *)up_pqs;
    const __global uint  *up_pqh_u = (const __global uint  *)(up_pqs + (size_t)num_blocks * 128);
    const __global uchar *up_ps    = up_pqs + (size_t)num_blocks * 192;
    const __global uchar *up_pd    = up_pqs + (size_t)num_blocks * 208;
    const __global uint  *gt_pql_u = (const __global uint  *)gt_pqs;
    const __global uint  *gt_pqh_u = (const __global uint  *)(gt_pqs + (size_t)num_blocks * 128);
    const __global uchar *gt_ps    = gt_pqs + (size_t)num_blocks * 192;
    const __global uchar *gt_pd    = gt_pqs + (size_t)num_blocks * 208;
#endif

    float up_acc = 0.0f, gate_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float *sb = sbase + (bid & 1u) * 256;

#if QCODE == MOE_SG_Q4_K
        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);

        uint8 uw0 = intel_sub_group_block_read8(up_pqs_u + pe_u +   0);
        uint8 uw1 = intel_sub_group_block_read8(up_pqs_u + pe_u + 128);
        uint8 uw2 = intel_sub_group_block_read8(up_pqs_u + pe_u + 256);
        uint8 uw3 = intel_sub_group_block_read8(up_pqs_u + pe_u + 384);
        uint8 gw0 = intel_sub_group_block_read8(gt_pqs_u + pe_u +   0);
        uint8 gw1 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 128);
        uint8 gw2 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 256);
        uint8 gw3 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 384);

        float isum[8], uscale[8], uminv[8], gscale[8], gminv[8];
        FUNC_CALL(sg_q4k_decode)(up_psl_pu, psl_pu_off, hh, uscale, uminv);
        FUNC_CALL(sg_q4k_decode)(gt_psl_pu, psl_pu_off, hh, gscale, gminv);
        FUNC_CALL(sg_q4k_load_inp)(sb, hidden_states + (size_t)t * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        up_acc   += FUNC_CALL(sg_q4k_acc)(uw0, uw1, uw2, uw3, uscale, uminv, isum, sb);
        gate_acc += FUNC_CALL(sg_q4k_acc)(gw0, gw1, gw2, gw3, gscale, gminv, isum, sb);
#elif QCODE == MOE_SG_Q5_K
        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e      = (h * nbpr + bid) * (OPG *  32 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG *  16 / 4);

        uint8 uw0 = intel_sub_group_block_read8(up_pqs_u + pe_u +   0);
        uint8 uw1 = intel_sub_group_block_read8(up_pqs_u + pe_u + 128);
        uint8 uw2 = intel_sub_group_block_read8(up_pqs_u + pe_u + 256);
        uint8 uw3 = intel_sub_group_block_read8(up_pqs_u + pe_u + 384);
        uint8 uhq = intel_sub_group_block_read8(up_pqh_u + pqh_e);
        uint8 gw0 = intel_sub_group_block_read8(gt_pqs_u + pe_u +   0);
        uint8 gw1 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 128);
        uint8 gw2 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 256);
        uint8 gw3 = intel_sub_group_block_read8(gt_pqs_u + pe_u + 384);
        uint8 ghq = intel_sub_group_block_read8(gt_pqh_u + pqh_e);

        float isum[8], uscale[8], uminv[8], gscale[8], gminv[8];
        FUNC_CALL(sg_q5k_decode)(up_psl_pu, psl_pu_off, hh, uscale, uminv);
        FUNC_CALL(sg_q5k_decode)(gt_psl_pu, psl_pu_off, hh, gscale, gminv);
        FUNC_CALL(sg_q5k_load_inp)(sb, hidden_states + (size_t)t * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        up_acc   += FUNC_CALL(sg_q5k_acc)(uw0, uw1, uw2, uw3, uhq, uscale, uminv, isum, sb);
        gate_acc += FUNC_CALL(sg_q5k_acc)(gw0, gw1, gw2, gw3, ghq, gscale, gminv, isum, sb);
#elif QCODE == MOE_SG_Q6_K
        const uint pql_e = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e = (h * nbpr + bid) * (OPG *  64 / 4);
        const uint ps_e  = (h * nbpr + bid) * (OPG * 16);
        const uint pd_e  = (h * nbpr + bid) * (OPG * 2);

        uint8 uw0 = intel_sub_group_block_read8(up_pql_u + pql_e +   0);
        uint8 uw1 = intel_sub_group_block_read8(up_pql_u + pql_e + 128);
        uint8 uw2 = intel_sub_group_block_read8(up_pql_u + pql_e + 256);
        uint8 uw3 = intel_sub_group_block_read8(up_pql_u + pql_e + 384);
        uint8 uh0 = intel_sub_group_block_read8(up_pqh_u + pqh_e +   0);
        uint8 uh1 = intel_sub_group_block_read8(up_pqh_u + pqh_e + 128);
        uchar16 uscq = intel_sub_group_block_read_uc16(up_ps + ps_e);
        ushort  udr  = intel_sub_group_block_read_us((const __global ushort *)(up_pd + pd_e));
        uint8 gw0 = intel_sub_group_block_read8(gt_pql_u + pql_e +   0);
        uint8 gw1 = intel_sub_group_block_read8(gt_pql_u + pql_e + 128);
        uint8 gw2 = intel_sub_group_block_read8(gt_pql_u + pql_e + 256);
        uint8 gw3 = intel_sub_group_block_read8(gt_pql_u + pql_e + 384);
        uint8 gh0 = intel_sub_group_block_read8(gt_pqh_u + pqh_e +   0);
        uint8 gh1 = intel_sub_group_block_read8(gt_pqh_u + pqh_e + 128);
        uchar16 gscq = intel_sub_group_block_read_uc16(gt_ps + ps_e);
        ushort  gdr  = intel_sub_group_block_read_us((const __global ushort *)(gt_pd + pd_e));

        uint8 iv = FUNC_CALL(sg_q6k_stage_inp)(sb, hidden_states + (size_t)t * input_len + (size_t)bid * 256, hh);
        float uclo[8], uchi[8], gclo[8], gchi[8];
        FUNC_CALL(sg_q6k_decode)(uscq, udr, uclo, uchi);
        FUNC_CALL(sg_q6k_decode)(gscq, gdr, gclo, gchi);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        float ucorr = FUNC_CALL(sg_q6k_corr)(iv, uclo, uchi);
        float gcorr = FUNC_CALL(sg_q6k_corr)(iv, gclo, gchi);
        up_acc   += FUNC_CALL(sg_q6k_acc)(uw0, uw1, uw2, uw3, uh0, uh1, uclo, uchi, sb) - 32.0f * ucorr;
        gate_acc += FUNC_CALL(sg_q6k_acc)(gw0, gw1, gw2, gw3, gh0, gh1, gclo, gchi, sb) - 32.0f * gcorr;
#endif
    }

#if KSPLIT == 1
    {
        float sig = 1.0f / (1.0f + native_exp(-gate_acc));
        gate_up_out[(size_t)t * output_num * output_len + (size_t)v * output_len + h * OPG + hh] =
            (half)((sig * gate_acc) * up_acc);
    }
#else
    slm_up  [sg * OPG + hh] = up_acc;
    slm_gate[sg * OPG + hh] = gate_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) {
            u  += slm_up  [r * OPG + hh];
            gt += slm_gate[r * OPG + hh];
        }
        float sig = 1.0f / (1.0f + native_exp(-gt));
        gate_up_out[(size_t)t * output_num * output_len + (size_t)v * output_len + h * OPG + hh] =
            (half)((sig * gt) * u);
    }
#endif
}

#endif  // GATE_UP_SG_ENABLE

// ============================================================================================
// ==========================  Kernel 2: down_merge_sg  ========================================
// Fuses the existing mlp_down + mlp_reduce stages: one work-group per (token t, output-row-
// group h) loops internally over all EXPERTS_PER_TOKEN routed experts of token t, accumulates
// efficient(=routing weight) * down(gate_up_out[t,e]) and writes the final hidden_states row
// directly (is_acc handling kept for parity with the reference / possible future shared-expert
// accumulation, but this integration always calls it with is_acc = 0; see
// moe_3gemm_swiglu_opt.cpp).
// global = (HIDDEN_SIZE, KSPLIT, token_num)
// local  = (OPG=16, KSPLIT, 1)
// ============================================================================================
#if DOWN_MERGE_SG_ENABLE

__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(down_merge_sg)(
    const __global half  * restrict gate_up_in,     /* [token_num*EXPERTS_PER_TOKEN, INTERMEDIATE_SIZE] */
    const __global uchar * restrict down_weights,   /* [EXPERT_NUM * expert_size] */
    const __global uint  * restrict expert_list,    /* [token_num, EXPERTS_PER_TOKEN] */
    const __global half  * restrict routing_weights,/* [token_num, EXPERTS_PER_TOKEN] */
          __global half  * restrict final_out,      /* [token_num, HIDDEN_SIZE] */
    int  is_acc
) {
    const uint input_num   = EXPERTS_PER_TOKEN;
    const uint input_len   = INTERMEDIATE_SIZE;
    const uint output_len  = HIDDEN_SIZE;
    const uint nbpr        = input_len / 256;
    const uint num_blocks  = input_len * output_len / 256;
    const uint expert_size = SG_EXPERT_BYTES(num_blocks);

    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float slm_red[KSPLIT * OPG];
    __local float *sbase = slm_inp + sg * (2 * 256);

    const uint total = input_num * nbpr;
#if DM_BLOCKED
    const uint per  = (total + KSPLIT - 1) / KSPLIT;
    const uint from = sg * per;
    const uint to   = min(from + per, total);
#endif
    float acc = 0.0f;

#if DM_BLOCKED
    for (uint idx = from; idx < to; idx++) {
#else
    for (uint idx = sg; idx < total; idx += KSPLIT) {
#endif
        const uint e   = idx / nbpr;
        const uint bid = idx - e * nbpr;
        __local float *sb = sbase + (idx & 1u) * 256;

        const uint  index     = expert_list[t * input_num + e];
        const float efficient = (float)routing_weights[t * input_num + e];

        const __global uchar *d_pqs = down_weights + (size_t)index * expert_size;

#if QCODE == MOE_SG_Q4_K
        const __global uint *pqs_u  = (const __global uint *)d_pqs;
        const __global uint *psl_pu = (const __global uint *)(d_pqs + (size_t)num_blocks * 128);

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG * 16 / 4);

        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);

        float isum[8], scale[8], minv[8];
        FUNC_CALL(sg_q4k_decode)(psl_pu, psl_pu_off, hh, scale, minv);
        FUNC_CALL(sg_q4k_load_inp)(sb, gate_up_in + (size_t)t * input_num * input_len
                        + (size_t)e * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        acc = fma(FUNC_CALL(sg_q4k_acc)(w0, w1, w2, w3, scale, minv, isum, sb), efficient, acc);
#elif QCODE == MOE_SG_Q5_K
        const __global uint *pqs_u  = (const __global uint *)d_pqs;
        const __global uint *pqh_u  = (const __global uint *)(d_pqs + (size_t)num_blocks * 128);
        const __global uint *psl_pu = (const __global uint *)(d_pqs + (size_t)num_blocks * 160);

        const uint pe_u       = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e      = (h * nbpr + bid) * (OPG *  32 / 4);
        const uint psl_pu_off = (h * nbpr + bid) * (OPG *  16 / 4);

        uint8 w0 = intel_sub_group_block_read8(pqs_u + pe_u +   0);
        uint8 w1 = intel_sub_group_block_read8(pqs_u + pe_u + 128);
        uint8 w2 = intel_sub_group_block_read8(pqs_u + pe_u + 256);
        uint8 w3 = intel_sub_group_block_read8(pqs_u + pe_u + 384);
        uint8 hq = intel_sub_group_block_read8(pqh_u + pqh_e);

        float isum[8], scale[8], minv[8];
        FUNC_CALL(sg_q5k_decode)(psl_pu, psl_pu_off, hh, scale, minv);
        FUNC_CALL(sg_q5k_load_inp)(sb, gate_up_in + (size_t)t * input_num * input_len
                        + (size_t)e * input_len + (size_t)bid * 256, hh, isum);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        acc = fma(FUNC_CALL(sg_q5k_acc)(w0, w1, w2, w3, hq, scale, minv, isum, sb), efficient, acc);
#elif QCODE == MOE_SG_Q6_K
        const __global uint  *pql_u = (const __global uint  *)d_pqs;
        const __global uint  *pqh_u = (const __global uint  *)(d_pqs + (size_t)num_blocks * 128);
        const __global uchar *ps    = d_pqs + (size_t)num_blocks * 192;
        const __global uchar *pd    = d_pqs + (size_t)num_blocks * 208;

        const uint pql_e = (h * nbpr + bid) * (OPG * 128 / 4);
        const uint pqh_e = (h * nbpr + bid) * (OPG *  64 / 4);
        const uint ps_e  = (h * nbpr + bid) * (OPG * 16);
        const uint pd_e  = (h * nbpr + bid) * (OPG * 2);

        uint8 w0  = intel_sub_group_block_read8(pql_u + pql_e +   0);
        uint8 w1  = intel_sub_group_block_read8(pql_u + pql_e + 128);
        uint8 w2  = intel_sub_group_block_read8(pql_u + pql_e + 256);
        uint8 w3  = intel_sub_group_block_read8(pql_u + pql_e + 384);
        uint8 hq0 = intel_sub_group_block_read8(pqh_u + pqh_e +   0);
        uint8 hq1 = intel_sub_group_block_read8(pqh_u + pqh_e + 128);
        uchar16 scq = intel_sub_group_block_read_uc16(ps + ps_e);
        ushort  dr  = intel_sub_group_block_read_us((const __global ushort *)(pd + pd_e));

        uint8 iv = FUNC_CALL(sg_q6k_stage_inp)(sb, gate_up_in + (size_t)t * input_num * input_len
                                    + (size_t)e * input_len + (size_t)bid * 256, hh);
        float clo[8], chi[8];
        FUNC_CALL(sg_q6k_decode)(scq, dr, clo, chi);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        float corr = FUNC_CALL(sg_q6k_corr)(iv, clo, chi);
        acc = fma(FUNC_CALL(sg_q6k_acc)(w0, w1, w2, w3, hq0, hq1, clo, chi, sb) - 32.0f * corr, efficient, acc);
#endif
    }

#if KSPLIT == 1
    {
        uint row = h * OPG + hh;
        float o = acc;
        if (is_acc) o += (float)final_out[(size_t)t * output_len + row];
        final_out[(size_t)t * output_len + row] = (half)o;
    }
#else
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float o = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) o += slm_red[r * OPG + hh];
        uint row = h * OPG + hh;
        if (is_acc) o += (float)final_out[(size_t)t * output_len + row];
        final_out[(size_t)t * output_len + row] = (half)o;
    }
#endif
}

#endif  // DOWN_MERGE_SG_ENABLE


// ============================================================================================
// ====================  Shared-expert Q8_0 kernels  ==========================================
// Standalone GEMV kernels for the (optional, single) shared expert when its gate/up/down
// weights are GGUF Q8_0.
//
// Unlike the routed Q4_K/Q5_K/Q6_K experts (which have their own dedicated SG byte layout,
// see the header comment at the top of this file), the shared expert's Q8_0 weights are now
// ALSO repacked at compile_model time -- by RepackGGUFMoEWeights::pack_shared_q8_0_sg() in
// repack_gguf_moe_weights.cpp -- into the SAME transposed "SG" (sub-group block-read) layout
// used by fc_gguf_q8_0_sg.cl / RepackGGUFWeightsShuffle's Q8_0 branch for plain FC. Concretely,
// for a [N, K] weight matrix with nbpr = K/256 super-blocks-per-row and OPG=16 lanes per
// row-group:
//   pqs section: N*nbpr*OPG*256 bytes at offset 0. Entry (h, bid) = (row_group, block-in-row)
//     at (h*nbpr+bid)*OPG*256 bytes; sub-block j (0..7) at +j*OPG*32, chunk c (0..7) at
//     +c*OPG*4 -- one intel_sub_group_block_read8 per sub-block delivers all 16 lanes' 32
//     signed weights (8 int8x4 chunks) for that sub-block in a single wide transaction.
//   pd  section: N*nbpr*OPG*16 bytes right after pqs. Entry (h, bid) at (h*nbpr+bid)*OPG*16
//     bytes, SoA fp16 field j at j*OPG*2 + lid*2 -- one intel_sub_group_block_read_us8 per
//     lane fetches all 8 sub-block scales.
// This lets shared_gate_up_q8_0 / shared_down_merge_q8_0 below use intel_sub_group_block_read8
// / intel_sub_group_block_read_us8 for BOTH the activations (as before) AND the weights
// themselves (new), eliminating the previous per-lane vload16 weight loads entirely -- the
// same technique fc_gguf_q8_0_sg.cl uses for the equivalent plain-FC Q8_0 GEMV.
//
//   shared_gate_scalar_q8_0: per-token scalar gate = sigmoid(dot(hidden_states[t,:],
//     shared_gate_vec)); shared_gate_vec is a plain (non-quantized) half[HIDDEN_SIZE] vector
//     (mirrors the assumption already made by the SHARED_EXPERT_ENABLE path in
//     moe_3gemm_swiglu_mlp.cl: "assume no scale/zp for now, or pre-dequantized"). Unaffected by
//     the SG repack (not a Q8_0 GEMV).
//     global = (SHARED_SCALAR_LWS, token_num), local = (SHARED_SCALAR_LWS, 1).
//   shared_gate_up_q8_0: gate_up_out[t, row] = silu(dot(x,gate_row)) * dot(x,up_row); same
//     SwiGLU convention as gate_up_sg, but a single fixed expert (no expert_list indirection).
//     global = (INTERMEDIATE_SIZE, KSPLIT, token_num), local = (OPG=16, KSPLIT, 1).
//   shared_down_merge_q8_0: final_out[t, row] (+)= shared_gate[t] * dot(gate_up_in[t,:],
//     down_row); this integration always calls it with is_acc=1 (the routed-expert sum from
//     down_merge_sg is written to final_out first).
//     global = (HIDDEN_SIZE, KSPLIT, token_num), local = (OPG=16, KSPLIT, 1).
// ============================================================================================

// ---------------------------------------------------------------------------------------------
// Sub-group block-read helpers shared by shared_gate_up_q8_0 / shared_down_merge_q8_0, mirroring
// fc_gguf_q8_0_sg.cl's ACC_CHUNK_Q8 / Q8_SUBBLOCK macros and activation-staging technique exactly.
//
// NOTE on why these MUST be named via FUNC()/FUNC_CALL() (like every other cross-KERNEL helper in
// this codebase, see the FUNC()/FUNC_CALL() helpers in moe_3gemm_swiglu_mlp.cl) and NOT protected
// by a plain `#ifndef ... #define ...` include guard:
//   This project's primitive db generator (primitive_db_gen.py: Kernels2CHeaders.append_undefs())
//   bakes, ONCE into the database string for this whole .cl file, an "#undef" for EVERY macro
//   `#define`d anywhere in the file (a blind textual scan, not scoped to any one KERNEL). That
//   generated undef block is part of kernel_string->str itself (see KernelBaseOpenCL::GetKernelString
//   in kernel_base_opencl.cpp), which is IDENTICAL and gets re-emitted in full once per KERNEL(...)
//   entry point compiled from this file (gate_up_sg / down_merge_sg / shared_gate_scalar_q8_0 /
//   shared_gate_up_q8_0 / shared_down_merge_q8_0), whenever kernels_cache batches 2+ of them into a
//   single clBuildProgram() call (full_code = kernel_string->jit + kernel_string->str +
//   kernel_string->undefs, concatenated back-to-back for every entry point in the batch -- see
//   kernels_cache::get_program_source()). So an include-guard macro defined inside `str` gets
//   #undef'd again at the END of that very same copy of `str` (by the auto-generated undef list),
//   and is therefore back to "not defined" by the time the next KERNEL's copy of `str` begins --
//   an `#ifndef` guard around a function body is silently defeated and the function is redefined,
//   which (unlike a redefined macro, which only warns) is a hard compile error.
//   The only construct in this file that reliably survives that per-copy undef reset is
//   FUNC(name)/FUNC_CALL(name) itself: FUNC/FUNC_CALL are redefined (to a *different*,
//   kernel_id-suffixed expansion) at the START of every copy by kernel_string->jit, so
//   `FUNC(sh_q8_0_decode_scales8)` expands to a distinct, non-colliding symbol name in every
//   KERNEL's copy of this file -- exactly like moe_gguf_gate_up_gemv & friends in
//   moe_3gemm_swiglu_mlp.cl. The macros below (SH_Q8_ACC_CHUNK / SH_Q8_SUBBLOCK) do NOT need this
//   treatment: redefining a macro is only ever a (harmless) warning, never an error.
// ---------------------------------------------------------------------------------------------

// Accumulate one 4-byte chunk (4 signed int8 weights) into a sub-block dot product. `in` points at
// the SLM base of the sub-block, `c` is the chunk index (0..7); weights c*4..c*4+3 map to
// activations in[c*4..c*4+3].
#define SH_Q8_ACC_CHUNK(dot, u, c, in)                                            \
    do {                                                                         \
        char4 _q = as_char4((uint)(u));                                          \
        (dot) = fma((in)[(c) * 4 + 0], (float)_q.s0, (dot));                     \
        (dot) = fma((in)[(c) * 4 + 1], (float)_q.s1, (dot));                     \
        (dot) = fma((in)[(c) * 4 + 2], (float)_q.s2, (dot));                     \
        (dot) = fma((in)[(c) * 4 + 3], (float)_q.s3, (dot));                     \
    } while (0)

// One SG-packed Q8_0 sub-block (32 signed weights, delivered as 8 uint chunks by ONE
// intel_sub_group_block_read8) dotted against 32 SLM-resident activations, scaled by the
// sub-block's f16 scale and accumulated into `acc`.
#define SH_Q8_SUBBLOCK(in, wv, sc, acc)                                                        \
    do {                                                                                       \
        float _dot = 0.0f;                                                                     \
        SH_Q8_ACC_CHUNK(_dot, (wv).s0, 0, in); SH_Q8_ACC_CHUNK(_dot, (wv).s1, 1, in);           \
        SH_Q8_ACC_CHUNK(_dot, (wv).s2, 2, in); SH_Q8_ACC_CHUNK(_dot, (wv).s3, 3, in);           \
        SH_Q8_ACC_CHUNK(_dot, (wv).s4, 4, in); SH_Q8_ACC_CHUNK(_dot, (wv).s5, 5, in);           \
        SH_Q8_ACC_CHUNK(_dot, (wv).s6, 6, in); SH_Q8_ACC_CHUNK(_dot, (wv).s7, 7, in);           \
        (acc) = fma((sc), _dot, (acc));                                                        \
    } while (0)

// Decode the 8 SoA fp16 sub-block scales fetched by one intel_sub_group_block_read_us8 per lane.
// Named via FUNC()/FUNC_CALL() -- see note above -- so each KERNEL's copy of this file gets its
// own uniquely-suffixed symbol and never collides with another copy in the same program build.
inline void FUNC(sh_q8_0_decode_scales8)(ushort8 dv, float *scale) {
    ushort dh[8];
    dh[0] = dv.s0; dh[1] = dv.s1; dh[2] = dv.s2; dh[3] = dv.s3;
    dh[4] = dv.s4; dh[5] = dv.s5; dh[6] = dv.s6; dh[7] = dv.s7;
    unroll_for (int i = 0; i < 8; i++)
        scale[i] = (float)as_half(dh[i]);
}

// Cooperative activation staging: ONE intel_sub_group_block_read8 fetches 256 contiguous half
// activations (8 native Q8_0 sub-blocks), distributed across the OPG=16 lanes' registers and
// re-assembled into a 256-float SLM buffer shared by every lane -- mirrors sg_q4k_load_inp /
// sg_q6k_stage_inp above (Q8_0 needs no isum/correction term: unlike Q4_K/Q5_K/Q6_K it has no
// zero-point/min, so a plain per-sub-block scale*dot is already exact). Named via FUNC()/
// FUNC_CALL() for the same cross-KERNEL-copy uniqueness reason as sh_q8_0_decode_scales8 above.
inline void FUNC(sh_q8_0_load_inp8)(__local float *sb, const __global half *src, uint hh) {
    uint8 iv = intel_sub_group_block_read8((const __global uint *)src);
#define SH_Q8_STAGE(j, comp) *(__local float2 *)(sb + (j) * 32 + 2 * hh) = convert_float2(as_half2(comp));
    SH_Q8_STAGE(0, iv.s0) SH_Q8_STAGE(1, iv.s1) SH_Q8_STAGE(2, iv.s2) SH_Q8_STAGE(3, iv.s3)
    SH_Q8_STAGE(4, iv.s4) SH_Q8_STAGE(5, iv.s5) SH_Q8_STAGE(6, iv.s6) SH_Q8_STAGE(7, iv.s7)
#undef SH_Q8_STAGE
}

#if SHARED_GATE_SCALAR_Q8_0_ENABLE

__attribute__((reqd_work_group_size(SHARED_SCALAR_LWS, 1, 1)))
KERNEL(shared_gate_scalar_q8_0)(
    const __global half * restrict hidden_states,   /* [token_num, HIDDEN_SIZE] */
    const __global half * restrict shared_gate_vec, /* [HIDDEN_SIZE] (plain, non-quantized) */
          __global half * restrict shared_gate_out  /* [token_num] */
) {
    const uint t   = get_group_id(1);
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);

    const __global half* x = hidden_states + (size_t)t * HIDDEN_SIZE;
    float partial = 0.0f;
    for (uint i = lid; i < HIDDEN_SIZE; i += lsz) {
        partial += (float)x[i] * (float)shared_gate_vec[i];
    }

    __local float slm[SHARED_SCALAR_LWS];
    slm[lid] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsz >> 1; s > 0; s >>= 1) {
        if (lid < s)
            slm[lid] += slm[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        shared_gate_out[t] = (half)(1.0f / (1.0f + native_exp(-slm[0])));
    }
}

#endif  // SHARED_GATE_SCALAR_Q8_0_ENABLE

#if SHARED_GATE_UP_Q8_0_ENABLE

// SG-packed Q8_0 GEMV, gate+up fused: one work-group covers OPG=16 output rows (h*OPG+hh) of
// BOTH the gate and up projections for one token; each sub-group (sg = K-split index) strides
// over the K-blocks (bid), issuing 16 intel_sub_group_block_read8 (8 gate + 8 up sub-blocks) +
// 2 intel_sub_group_block_read_us8 (gate + up scales) + 1 intel_sub_group_block_read8 (shared
// activation window) per iteration -- no per-lane scalar/vload weight or activation loads at all.
__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(shared_gate_up_q8_0)(
    const __global uchar * restrict gate_weights,  /* SG-packed Q8_0 [INTERMEDIATE_SIZE, HIDDEN_SIZE] */
    const __global uchar * restrict up_weights,    /* SG-packed Q8_0 [INTERMEDIATE_SIZE, HIDDEN_SIZE] */
    const __global half  * restrict hidden_states, /* [token_num, HIDDEN_SIZE] */
          __global half  * restrict gate_up_out    /* [token_num, INTERMEDIATE_SIZE] */
) {
    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint t  = get_group_id(2);

    const uint nbpr   = HIDDEN_SIZE / 256u;               // 256-elem super-blocks per row
    const uint off_pd = (uint)INTERMEDIATE_SIZE * nbpr * 256u;

    const __global uint*   g_pqs_u = (const __global uint*)gate_weights;
    const __global ushort* g_pd_us = (const __global ushort*)(gate_weights + off_pd);
    const __global uint*   u_pqs_u = (const __global uint*)up_weights;
    const __global ushort* u_pd_us = (const __global ushort*)(up_weights + off_pd);

    const __global half* x = hidden_states + (size_t)t * HIDDEN_SIZE;

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float* sbase = slm_inp + sg * (2 * 256);

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;

        // Issue all long-latency global loads first: 8 sub-block chunks x 2 projections + the
        // 2 scale vectors -- each a SINGLE wide sub-group block read serving all 16 lanes at once.
        const uint pe_u = (h * nbpr + bid) * (OPG * 256u / 4u);
        uint8 gw0 = intel_sub_group_block_read8(g_pqs_u + pe_u + 0 * OPG * 8);
        uint8 gw1 = intel_sub_group_block_read8(g_pqs_u + pe_u + 1 * OPG * 8);
        uint8 gw2 = intel_sub_group_block_read8(g_pqs_u + pe_u + 2 * OPG * 8);
        uint8 gw3 = intel_sub_group_block_read8(g_pqs_u + pe_u + 3 * OPG * 8);
        uint8 gw4 = intel_sub_group_block_read8(g_pqs_u + pe_u + 4 * OPG * 8);
        uint8 gw5 = intel_sub_group_block_read8(g_pqs_u + pe_u + 5 * OPG * 8);
        uint8 gw6 = intel_sub_group_block_read8(g_pqs_u + pe_u + 6 * OPG * 8);
        uint8 gw7 = intel_sub_group_block_read8(g_pqs_u + pe_u + 7 * OPG * 8);
        uint8 uw0 = intel_sub_group_block_read8(u_pqs_u + pe_u + 0 * OPG * 8);
        uint8 uw1 = intel_sub_group_block_read8(u_pqs_u + pe_u + 1 * OPG * 8);
        uint8 uw2 = intel_sub_group_block_read8(u_pqs_u + pe_u + 2 * OPG * 8);
        uint8 uw3 = intel_sub_group_block_read8(u_pqs_u + pe_u + 3 * OPG * 8);
        uint8 uw4 = intel_sub_group_block_read8(u_pqs_u + pe_u + 4 * OPG * 8);
        uint8 uw5 = intel_sub_group_block_read8(u_pqs_u + pe_u + 5 * OPG * 8);
        uint8 uw6 = intel_sub_group_block_read8(u_pqs_u + pe_u + 6 * OPG * 8);
        uint8 uw7 = intel_sub_group_block_read8(u_pqs_u + pe_u + 7 * OPG * 8);

        const uint pd_off = (h * nbpr + bid) * (OPG * 8u);
        ushort8 gdv = intel_sub_group_block_read_us8(g_pd_us + pd_off);
        ushort8 udv = intel_sub_group_block_read_us8(u_pd_us + pd_off);

        // Cooperative activation staging (ONE wide read for all 16 lanes), overlaps the loads above.
        FUNC_CALL(sh_q8_0_load_inp8)(sb, x + (size_t)bid * 256, hh);

        float gscale[8], uscale[8];
        FUNC_CALL(sh_q8_0_decode_scales8)(gdv, gscale);
        FUNC_CALL(sh_q8_0_decode_scales8)(udv, uscale);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        SH_Q8_SUBBLOCK(sb + 0 * 32, gw0, gscale[0], gate_acc); SH_Q8_SUBBLOCK(sb + 0 * 32, uw0, uscale[0], up_acc);
        SH_Q8_SUBBLOCK(sb + 1 * 32, gw1, gscale[1], gate_acc); SH_Q8_SUBBLOCK(sb + 1 * 32, uw1, uscale[1], up_acc);
        SH_Q8_SUBBLOCK(sb + 2 * 32, gw2, gscale[2], gate_acc); SH_Q8_SUBBLOCK(sb + 2 * 32, uw2, uscale[2], up_acc);
        SH_Q8_SUBBLOCK(sb + 3 * 32, gw3, gscale[3], gate_acc); SH_Q8_SUBBLOCK(sb + 3 * 32, uw3, uscale[3], up_acc);
        SH_Q8_SUBBLOCK(sb + 4 * 32, gw4, gscale[4], gate_acc); SH_Q8_SUBBLOCK(sb + 4 * 32, uw4, uscale[4], up_acc);
        SH_Q8_SUBBLOCK(sb + 5 * 32, gw5, gscale[5], gate_acc); SH_Q8_SUBBLOCK(sb + 5 * 32, uw5, uscale[5], up_acc);
        SH_Q8_SUBBLOCK(sb + 6 * 32, gw6, gscale[6], gate_acc); SH_Q8_SUBBLOCK(sb + 6 * 32, uw6, uscale[6], up_acc);
        SH_Q8_SUBBLOCK(sb + 7 * 32, gw7, gscale[7], gate_acc); SH_Q8_SUBBLOCK(sb + 7 * 32, uw7, uscale[7], up_acc);
    }

    const uint row = h * OPG + hh;
#if KSPLIT == 1
    {
        float sig = 1.0f / (1.0f + native_exp(-gate_acc));
        gate_up_out[(size_t)t * INTERMEDIATE_SIZE + row] = (half)((sig * gate_acc) * up_acc);
    }
#else
    __local float slm_up[KSPLIT * OPG];
    __local float slm_gate[KSPLIT * OPG];
    slm_up[sg * OPG + hh]   = up_acc;
    slm_gate[sg * OPG + hh] = gate_acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float u = 0.0f, gt = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) {
            u  += slm_up[r * OPG + hh];
            gt += slm_gate[r * OPG + hh];
        }
        float sig = 1.0f / (1.0f + native_exp(-gt));
        gate_up_out[(size_t)t * INTERMEDIATE_SIZE + row] = (half)((sig * gt) * u);
    }
#endif
}

#endif  // SHARED_GATE_UP_Q8_0_ENABLE

#if SHARED_DOWN_MERGE_Q8_0_ENABLE

// SG-packed Q8_0 GEMV, down projection: same technique as shared_gate_up_q8_0 above, single
// projection. is_acc selects overwrite (0) vs accumulate-onto-existing-output (1); this
// integration always calls it with is_acc=1 (the routed-expert sum from down_merge_sg /
// mlp_reduce is written to final_out first -- see moe_3gemm_swiglu_opt.cpp exec_batched_gemv()).
__attribute__((intel_reqd_sub_group_size(OPG)))
KERNEL(shared_down_merge_q8_0)(
    const __global half  * restrict gate_up_in,   /* [token_num, INTERMEDIATE_SIZE] */
    const __global uchar * restrict down_weights, /* SG-packed Q8_0 [HIDDEN_SIZE, INTERMEDIATE_SIZE] */
    const __global half  * restrict shared_gate,  /* [token_num] scalar gate */
          __global half  * restrict final_out,    /* [token_num, HIDDEN_SIZE] */
    int  is_acc
) {
    const uint h   = get_group_id(0);
    const uint hh  = get_local_id(0);
    const uint sg  = get_local_id(1);
    const uint t   = get_group_id(2);

    const uint nbpr   = INTERMEDIATE_SIZE / 256u;
    const uint off_pd = (uint)HIDDEN_SIZE * nbpr * 256u;

    const __global uint*   d_pqs_u = (const __global uint*)down_weights;
    const __global ushort* d_pd_us = (const __global ushort*)(down_weights + off_pd);

    const __global half* x = gate_up_in + (size_t)t * INTERMEDIATE_SIZE;

    __local float slm_inp[KSPLIT * 2 * 256];
    __local float* sbase = slm_inp + sg * (2 * 256);

    float acc = 0.0f;

    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        __local float* sb = sbase + (bid & 1u) * 256;

        const uint pe_u = (h * nbpr + bid) * (OPG * 256u / 4u);
        uint8 w0 = intel_sub_group_block_read8(d_pqs_u + pe_u + 0 * OPG * 8);
        uint8 w1 = intel_sub_group_block_read8(d_pqs_u + pe_u + 1 * OPG * 8);
        uint8 w2 = intel_sub_group_block_read8(d_pqs_u + pe_u + 2 * OPG * 8);
        uint8 w3 = intel_sub_group_block_read8(d_pqs_u + pe_u + 3 * OPG * 8);
        uint8 w4 = intel_sub_group_block_read8(d_pqs_u + pe_u + 4 * OPG * 8);
        uint8 w5 = intel_sub_group_block_read8(d_pqs_u + pe_u + 5 * OPG * 8);
        uint8 w6 = intel_sub_group_block_read8(d_pqs_u + pe_u + 6 * OPG * 8);
        uint8 w7 = intel_sub_group_block_read8(d_pqs_u + pe_u + 7 * OPG * 8);

        const uint pd_off = (h * nbpr + bid) * (OPG * 8u);
        ushort8 dv = intel_sub_group_block_read_us8(d_pd_us + pd_off);

        FUNC_CALL(sh_q8_0_load_inp8)(sb, x + (size_t)bid * 256, hh);

        float scale[8];
        FUNC_CALL(sh_q8_0_decode_scales8)(dv, scale);

        sub_group_barrier(CLK_LOCAL_MEM_FENCE);

        SH_Q8_SUBBLOCK(sb + 0 * 32, w0, scale[0], acc);
        SH_Q8_SUBBLOCK(sb + 1 * 32, w1, scale[1], acc);
        SH_Q8_SUBBLOCK(sb + 2 * 32, w2, scale[2], acc);
        SH_Q8_SUBBLOCK(sb + 3 * 32, w3, scale[3], acc);
        SH_Q8_SUBBLOCK(sb + 4 * 32, w4, scale[4], acc);
        SH_Q8_SUBBLOCK(sb + 5 * 32, w5, scale[5], acc);
        SH_Q8_SUBBLOCK(sb + 6 * 32, w6, scale[6], acc);
        SH_Q8_SUBBLOCK(sb + 7 * 32, w7, scale[7], acc);
    }

    const float gscale = (float)shared_gate[t];
    const uint row = h * OPG + hh;

#if KSPLIT == 1
    {
        float o = acc * gscale;
        if (is_acc)
            o += (float)final_out[(size_t)t * HIDDEN_SIZE + row];
        final_out[(size_t)t * HIDDEN_SIZE + row] = (half)o;
    }
#else
    __local float slm_red[KSPLIT * OPG];
    slm_red[sg * OPG + hh] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0) {
        float o = 0.0f;
        unroll_for (int r = 0; r < KSPLIT; r++) o += slm_red[r * OPG + hh];
        o *= gscale;
        if (is_acc)
            o += (float)final_out[(size_t)t * HIDDEN_SIZE + row];
        final_out[(size_t)t * HIDDEN_SIZE + row] = (half)o;
    }
#endif
}

#endif  // SHARED_DOWN_MERGE_Q8_0_ENABLE
