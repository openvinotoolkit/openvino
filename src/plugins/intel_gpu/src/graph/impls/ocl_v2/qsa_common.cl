// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Shared helpers of the QSA indexer / sparse attention OpenCL kernels.
//
// Vector elements are spread over the subgroup lanes: element `lane + SG * j` of a row lives in slot `j` of lane
// `lane` (the layout produced by intel_sub_group_block_read_us*).
//
// Every helper is decorated with FUNC() so that the kernels batched into one program do not redefine it,
// and the header deliberately has no include guard (each kernel gets its own copy, built against its own JIT).
//
// Opt-in sections (define before including):
//   QSA_INDEXER_HELPERS   - RMSNorm / partial NEOX RoPE of an index head vector (needs QSA_IDX_HEAD_DIM, ...)
//   QSA_ATTENTION_HELPERS - paged / dense input signature, K/V row addressing, scalar attend and, with
//                           QSA_XMX_WORKERS, the XMX building blocks (needs QSA_K_HEAD_SIZE, ...)

#include "include/batch_headers/common.cl"

#ifdef cl_intel_subgroups_short
#    pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#endif
#ifdef cl_intel_subgroups
#    pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif
#ifdef cl_intel_subgroup_local_block_io
#    pragma OPENCL EXTENSION cl_intel_subgroup_local_block_io : enable
#endif

#ifndef QSA_MAX_NE
#    define QSA_MAX_NE 32
#endif
#define QSA_SG SUBGROUP_SIZE
#define QSA_NEG_INF (-3.0e38f)
#define QSA_LOG2E 1.4426950408889634f

// ---------------------------------------------------------------------------
// Subgroup block IO: out[j] / v[j] is element (lane + SG * j) of the row
// ---------------------------------------------------------------------------
inline void FUNC(qsa_rd)(const __global ushort* p, __private ushort* out, const uint n) {
    uint j = 0;
    for (; j + 8 <= n; j += 8) {
        ushort8 r = intel_sub_group_block_read_us8(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1; out[j + 2] = r.s2; out[j + 3] = r.s3;
        out[j + 4] = r.s4; out[j + 5] = r.s5; out[j + 6] = r.s6; out[j + 7] = r.s7;
    }
    if (n - j >= 4) {
        ushort4 r = intel_sub_group_block_read_us4(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1; out[j + 2] = r.s2; out[j + 3] = r.s3;
        j += 4;
    }
    if (n - j >= 2) {
        ushort2 r = intel_sub_group_block_read_us2(p + j * QSA_SG);
        out[j + 0] = r.s0; out[j + 1] = r.s1;
        j += 2;
    }
    if (n - j >= 1) {
        out[j] = intel_sub_group_block_read_us(p + j * QSA_SG);
    }
}

inline void FUNC(qsa_wr)(__global ushort* p, const __private ushort* v, const uint n) {
    uint j = 0;
    for (; j + 8 <= n; j += 8) {
        intel_sub_group_block_write_us8(p + j * QSA_SG, (ushort8)(v[j + 0], v[j + 1], v[j + 2], v[j + 3], v[j + 4], v[j + 5], v[j + 6], v[j + 7]));
    }
    if (n - j >= 4) {
        intel_sub_group_block_write_us4(p + j * QSA_SG, (ushort4)(v[j + 0], v[j + 1], v[j + 2], v[j + 3]));
        j += 4;
    }
    if (n - j >= 2) {
        intel_sub_group_block_write_us2(p + j * QSA_SG, (ushort2)(v[j + 0], v[j + 1]));
        j += 2;
    }
    if (n - j >= 1) {
        intel_sub_group_block_write_us(p + j * QSA_SG, v[j]);
    }
}

// f16 row -> f32
inline void FUNC(qsa_rd_f32)(const __global half* p, __private float* v, const uint n) {
    ushort t[QSA_MAX_NE];
    FUNC_CALL(qsa_rd)((const __global ushort*)p, t, n);
    unroll_for (uint j = 0; j < n; j++) {
        v[j] = convert_float(as_half(t[j]));
    }
}

// f32 row -> f16 row (RTNE)
inline void FUNC(qsa_wr_f32)(__global half* p, const __private float* v, const uint n) {
    ushort t[QSA_MAX_NE];
    unroll_for (uint j = 0; j < n; j++) {
        t[j] = as_ushort(convert_half_rte(v[j]));
    }
    FUNC_CALL(qsa_wr)((__global ushort*)p, t, n);
}

// ---------------------------------------------------------------------------
// Address arithmetic (element offsets)
// ---------------------------------------------------------------------------
inline uint FUNC(qsa_kv_cache_offset)(uint page, uint kv_head, uint token_in_page, uint kv_heads, uint pa_block_size, uint head_size) {
    return ((page * kv_heads + kv_head) * pa_block_size + token_in_page) * head_size;
}

inline uint FUNC(qsa_num_complete_blocks)(uint abs_pos) {
    return (abs_pos + 1) / (uint)QSA_COMPRESS_RATIO;
}

// Resolves a flattened paged token to its sequence and absolute position; false for rows past the last real token.
inline bool FUNC(qsa_resolve_token_paged)(const __global int* subsequence_begins,
                                          const __global int* past_lens,
                                          uint num_seqs,
                                          uint token_idx,
                                          __private uint* seq,
                                          __private uint* abs_pos) {
    *seq = 0;
    *abs_pos = 0;
    for (uint i = 0; i < num_seqs; i++) {
        if (token_idx >= (uint)subsequence_begins[i] && token_idx < (uint)subsequence_begins[i + 1]) {
            *seq = i;
            *abs_pos = (uint)past_lens[i] + token_idx - (uint)subsequence_begins[i];
            return true;
        }
    }
    return false;
}

// Maps query tile `tile` onto its sequence and the global index of its first token.
inline bool FUNC(qsa_resolve_tile_paged)(const __global int* subsequence_begins,
                                         uint num_seqs,
                                         uint tile,
                                         uint tile_tokens,
                                         __private uint* seq,
                                         __private uint* token_begin) {
    uint acc = 0;
    for (uint i = 0; i < num_seqs; i++) {
        const uint begin = (uint)subsequence_begins[i];
        const uint q_len = (uint)subsequence_begins[i + 1] - begin;
        const uint n = (q_len + tile_tokens - 1) / tile_tokens;
        if (tile < acc + n) {
            *seq = i;
            *token_begin = begin + (tile - acc) * tile_tokens;
            return true;
        }
        acc += n;
    }
    return false;
}

// Dense token t = b * seq_len + l -> batch and logical position; a left-padding row yields 0xFFFFFFFF.
inline void FUNC(qsa_resolve_token_dense)(const __global int* key_length, uint seq_len, uint token_idx, __private uint* batch, __private uint* abs_pos) {
    *batch = token_idx / seq_len;
    const uint shifted = (uint)key_length[*batch] + (token_idx - *batch * seq_len);
    *abs_pos = shifted >= seq_len ? shifted - seq_len : 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------
// One online-softmax step on a score that is uniform over the subgroup.
inline void FUNC(qsa_softmax_step)(float score, __private float* running_max, __private float* running_sum, __private float* corr, __private float* weight) {
    const float new_max = *running_max > score ? *running_max : score;
    const float e0 = native_exp2((*running_max - new_max) * QSA_LOG2E);
    const float e1 = native_exp2((score - new_max) * QSA_LOG2E);
    *corr = e0;
    *weight = e1;
    *running_sum = *running_sum * e0 + e1;
    *running_max = new_max;
}

// ---------------------------------------------------------------------------
// Indexer-only helpers: RMSNorm and partial NEOX RoPE over a head vector spread over the subgroup lanes.
// ---------------------------------------------------------------------------
#ifdef QSA_INDEXER_HELPERS
#    define QSA_NE (QSA_IDX_HEAD_DIM / QSA_SG)

// RMSNorm with f32 accumulation; the gain is applied exactly as stored.
inline void FUNC(qsa_rms_norm)(__private float* v, const __global half* weight) {
    const uint lane = get_sub_group_local_id();
    float sumsq = 0.0f;
    unroll_for (uint j = 0; j < QSA_NE; j++) {
        sumsq += v[j] * v[j];
    }
    sumsq = sub_group_reduce_add(sumsq);
    const float denom = sumsq / (float)QSA_IDX_HEAD_DIM + (float)QSA_IDX_NORM_EPS;
    const float inv = native_rsqrt(denom);
    unroll_for (uint j = 0; j < QSA_NE; j++) {
        v[j] = v[j] * inv * convert_float(weight[lane + QSA_SG * j]);
    }
}

#    if QSA_IDX_ROTARY_DIM > 0
// Partial NEOX rope: channel d < H pairs with d + H (H = rotary_dim / 2). `cp[d * cs]` / `sp[d * cs]` are the coefficients.
inline void FUNC(qsa_rope)(__private float* v, const __global half* cp, const __global half* sp, const uint cs) {
    const uint lane = get_sub_group_local_id();
    const uint H = QSA_IDX_ROTARY_DIM / 2;
    float o[QSA_NE];
    unroll_for (uint j = 0; j < QSA_NE; j++) {
        const uint d = lane + QSA_SG * j;
        float res = v[j];
#        if ((QSA_IDX_ROTARY_DIM / 2) % SUBGROUP_SIZE) == 0
        const uint hs = H / QSA_SG;
        float pv = 0.0f;
        if (QSA_SG * j < QSA_IDX_ROTARY_DIM) {
            pv = (QSA_SG * j < H) ? v[min(j + hs, (uint)(QSA_NE - 1))] : v[j >= hs ? j - hs : 0];
        }
#        else
        const uint dp = d < H ? d + H : (d < QSA_IDX_ROTARY_DIM ? d - H : d);
        const uint lp = dp % QSA_SG;
        const uint jp = dp / QSA_SG;
        float pv = 0.0f;
        unroll_for (uint jj = 0; jj < QSA_NE; jj++) {
            const float t = intel_sub_group_shuffle(v[jj], lp);
            pv = jp == jj ? t : pv;
        }
#        endif
        if (d < QSA_IDX_ROTARY_DIM) {
            const float c = convert_float(cp[d * cs]);
            const float s = convert_float(sp[d * cs]);
            res = d < H ? v[j] * c - pv * s : v[j] * c + pv * s;
        }
        o[j] = res;
    }
    unroll_for (uint j = 0; j < QSA_NE; j++) {
        v[j] = o[j];
    }
}
#    endif
#endif  // QSA_INDEXER_HELPERS

// ---------------------------------------------------------------------------
// Shared pieces of the sparse attention kernels: the paged / dense (QSA_ADDRESS_MODE 0 / 1) input signature, the K/V
// row addressing, the scalar attend helper and (when QSA_XMX_WORKERS is defined before including) the XMX building
// blocks of the systolic kernels.
//
// Kernels using QSA_KROW / QSA_VROW must declare, in paged mode, `block_indices`, `page_begin` and `kv_head`, and in
// dense mode `batch`, `kv_head`, `shift` (history_len - key_length), `k_pitch`, `v_pitch`.
// ---------------------------------------------------------------------------
#ifdef QSA_ATTENTION_HELPERS
#    ifndef QSA_ADDRESS_MODE
#        define QSA_ADDRESS_MODE 0
#    endif

#    if QSA_ADDRESS_MODE == 0
#        define QSA_ATT_INPUT_ARGS const __global half* query, const __global half* key_cache, const __global half* value_cache, const __global int* past_lens, const __global int* subsequence_begins, const __global int* block_indices, const __global int* block_indices_begins, const __global int* selected_blocks, const __global int* selected_counts
#        define QSA_KROW(pos) FUNC_CALL(qsa_kv_cache_offset)((uint)block_indices[page_begin + (pos) / QSA_PAGE_SIZE], kv_head, (pos) % QSA_PAGE_SIZE, QSA_KV_HEADS, QSA_PAGE_SIZE, QSA_K_HEAD_SIZE)
#        define QSA_VROW(pos) FUNC_CALL(qsa_kv_cache_offset)((uint)block_indices[page_begin + (pos) / QSA_PAGE_SIZE], kv_head, (pos) % QSA_PAGE_SIZE, QSA_KV_HEADS, QSA_PAGE_SIZE, QSA_V_HEAD_SIZE)
#    else
#        define QSA_ATT_INPUT_ARGS const __global half* query, const __global half* key_cache, const __global half* value_cache, const __global int* selected_blocks, const __global int* selected_counts, const __global int* key_length
#        define QSA_KROW(pos) (((((batch) * QSA_KV_HEADS) + kv_head) * k_pitch + shift + (pos)) * QSA_K_HEAD_SIZE)
#        define QSA_VROW(pos) (((((batch) * QSA_KV_HEADS) + kv_head) * v_pitch + shift + (pos)) * QSA_V_HEAD_SIZE)
#    endif

#    define QSA_NEK (QSA_K_HEAD_SIZE / QSA_SG)
#    define QSA_NEV (QSA_V_HEAD_SIZE / QSA_SG)

inline void FUNC(qsa_scalar_attend)(const __global half* krow,
                                    const __global half* vrow,
                                    __private const float* q,
                                    __private float* acc,
                                    __private float* running_max,
                                    __private float* running_sum) {
    ushort kr[QSA_NEK];
    ushort vr[QSA_NEV];
    FUNC_CALL(qsa_rd)((const __global ushort*)krow, kr, QSA_NEK);
    FUNC_CALL(qsa_rd)((const __global ushort*)vrow, vr, QSA_NEV);

    float dot = 0.0f;
    unroll_for (uint j = 0; j < QSA_NEK; j++) {
        dot += q[j] * convert_float(as_half(kr[j]));
    }
    const float score = sub_group_reduce_add(dot);

    float corr = 0.0f, weight = 0.0f;
    FUNC_CALL(qsa_softmax_step)(score, running_max, running_sum, &corr, &weight);
    unroll_for (uint j = 0; j < QSA_NEV; j++) {
        acc[j] = acc[j] * corr + convert_float(as_half(vr[j])) * weight;
    }
}

#    ifdef QSA_XMX_WORKERS
// ---------------------------------------------------------------------------
// XMX building blocks (SIMD16). One KV step = 16 rows, row i of the step is held by lane i; the lane axis of the
// score tile is the query column.
// ---------------------------------------------------------------------------
#        define QSA_XW QSA_XMX_WORKERS
#        define QSA_XDH (QSA_K_HEAD_SIZE / QSA_XW)
#        define QSA_XDV (QSA_V_HEAD_SIZE / QSA_XW)
#        define QSA_XKT (QSA_XDH / 16)
#        define QSA_XVT (QSA_XDV / 8)
#        define QSA_XST_FLOATS (2 * 8 * QSA_SG)

#        ifdef cl_intel_subgroup_local_block_io
#            define QSA_SLM_WRITE8(p, v) intel_sub_group_block_write8((__local uint*)(p), as_uint8(v))
#            define QSA_SLM_READ8(p) as_float8(intel_sub_group_block_read8((__local uint*)(p)))
#        else
#            define QSA_SLM_WRITE8(p, v) FUNC_CALL(qsa_slm_write8)((__local float*)(p), v)
#            define QSA_SLM_READ8(p) FUNC_CALL(qsa_slm_read8)((__local float*)(p))
inline void FUNC(qsa_slm_write8)(__local float* p, float8 v) {
    const uint l = get_sub_group_local_id();
    p[0 * QSA_SG + l] = v.s0; p[1 * QSA_SG + l] = v.s1; p[2 * QSA_SG + l] = v.s2; p[3 * QSA_SG + l] = v.s3;
    p[4 * QSA_SG + l] = v.s4; p[5 * QSA_SG + l] = v.s5; p[6 * QSA_SG + l] = v.s6; p[7 * QSA_SG + l] = v.s7;
}
inline float8 FUNC(qsa_slm_read8)(__local float* p) {
    const uint l = get_sub_group_local_id();
    return (float8)(p[0 * QSA_SG + l], p[1 * QSA_SG + l], p[2 * QSA_SG + l], p[3 * QSA_SG + l],
                    p[4 * QSA_SG + l], p[5 * QSA_SG + l], p[6 * QSA_SG + l], p[7 * QSA_SG + l]);
}
#        endif

// K rows of one step as the A operand: kr[i][k] = K[row i][worker * XDH + 16 k + lane]. `koff` is this lane's
// (row = lane) element offset into the K cache; rows >= valid_kv are zero.
inline void FUNC(qsa_xmx_k_load)(const __global half* key_cache, const uint koff, const uint valid_kv, const uint worker, __private ushort (*kr)[QSA_XKT]) {
    unroll_for (uint i = 0; i < 16; i++) {
        const uint ko = sub_group_broadcast(koff, i);
        if (i < valid_kv) {
            FUNC_CALL(qsa_rd)((const __global ushort*)(key_cache + ko + worker * QSA_XDH), kr[i], QSA_XKT);
        } else {
            unroll_for (uint k = 0; k < QSA_XKT; k++) {
                kr[i][k] = 0;
            }
        }
    }
}

// QK of one step from preloaded K rows: st[i] = sum over the worker slices of K[row i] . q (this lane's column). With
// several workers the partial tiles are summed through SLM in the fixed order w = 0..W-1, so every worker sees
// bit-identical scores.
inline void FUNC(qsa_xmx_qk_regs)(__private ushort (*kr)[QSA_XKT],
                                  __private const int8* rq,
                                  const uint worker,
#        if QSA_XW > 1
                                  __local float* slm,  // [W][2][8][SG] of the current slot
#        endif
                                  __private float* st) {
    float8 st0 = (float8)(0.0f);
    float8 st1 = (float8)(0.0f);
    unroll_for (uint k = 0; k < QSA_XKT; k++) {
        const short8 a0 = (short8)((short)kr[0][k], (short)kr[1][k], (short)kr[2][k], (short)kr[3][k],
                                   (short)kr[4][k], (short)kr[5][k], (short)kr[6][k], (short)kr[7][k]);
        const short8 a1 = (short8)((short)kr[8][k], (short)kr[9][k], (short)kr[10][k], (short)kr[11][k],
                                   (short)kr[12][k], (short)kr[13][k], (short)kr[14][k], (short)kr[15][k]);
        st0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, rq[k], st0);
        st1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, rq[k], st1);
    }
#        if QSA_XW > 1
    QSA_SLM_WRITE8(slm + worker * QSA_XST_FLOATS, st0);
    QSA_SLM_WRITE8(slm + worker * QSA_XST_FLOATS + 8 * QSA_SG, st1);
    barrier(CLK_LOCAL_MEM_FENCE);
    st0 = QSA_SLM_READ8(slm);
    st1 = QSA_SLM_READ8(slm + 8 * QSA_SG);
    for (uint w = 1; w < QSA_XW; w++) {
        st0 += QSA_SLM_READ8(slm + w * QSA_XST_FLOATS);
        st1 += QSA_SLM_READ8(slm + w * QSA_XST_FLOATS + 8 * QSA_SG);
    }
#        endif
    vstore8(st0, 0, st);
    vstore8(st1, 0, st + 8);
}

inline void FUNC(qsa_xmx_qk)(const __global half* key_cache,
                             const uint koff,
                             const uint valid_kv,
                             __private const int8* rq,
                             const uint worker,
#        if QSA_XW > 1
                             __local float* slm,
#        endif
                             __private float* st) {
    ushort kr[16][QSA_XKT];
    FUNC_CALL(qsa_xmx_k_load)(key_cache, koff, valid_kv, worker, kr);
#        if QSA_XW > 1
    FUNC_CALL(qsa_xmx_qk_regs)(kr, rq, worker, slm, st);
#        else
    FUNC_CALL(qsa_xmx_qk_regs)(kr, rq, worker, st);
#        endif
}

// This lane's (row = lane) slice of V for one step: va[d] = V[row][worker * XDV + d * 8 .. + 8], zero when !lv.
// Rows are DV-aligned, so 32-byte vector loads are legal (scalar half loads would split into dword gathers).
inline void FUNC(qsa_xmx_v_load)(const __global half* value_cache, const uint voff, const bool lv, const uint worker, __private uint4* va) {
    const __global uint8* src = (const __global uint8*)(value_cache + voff + worker * QSA_XDV);
    unroll_for (uint d = 0; d < QSA_XVT; d += 2) {
        const uint8 v = src[d / 2];
        va[d] = lv ? v.lo : (uint4)(0);
        va[d + 1] = lv ? v.hi : (uint4)(0);
    }
}

// PV of one step: ro[d][i] += sum over the 16 rows of V[row][dv0 + i] * p[row] (this lane's column). V^T is the A
// operand (one KV row per lane, preloaded by qsa_xmx_v_load), P^T the B operand packed in-lane.
inline void FUNC(qsa_xmx_pv_regs)(__private const uint4* va, __private const float* p, __private float8* ro) {
    const int8 bp = (int8)(as_int(convert_half2_rte((float2)(p[0], p[1]))), as_int(convert_half2_rte((float2)(p[2], p[3]))),
                           as_int(convert_half2_rte((float2)(p[4], p[5]))), as_int(convert_half2_rte((float2)(p[6], p[7]))),
                           as_int(convert_half2_rte((float2)(p[8], p[9]))), as_int(convert_half2_rte((float2)(p[10], p[11]))),
                           as_int(convert_half2_rte((float2)(p[12], p[13]))), as_int(convert_half2_rte((float2)(p[14], p[15]))));
    unroll_for (uint d = 0; d < QSA_XVT; d++) {
        ro[d] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(va[d]), bp, ro[d]);
    }
}

inline void FUNC(qsa_xmx_pv)(const __global half* value_cache,
                             const uint voff,
                             const bool lv,
                             __private const float* p,
                             const uint worker,
                             __private float8* ro) {
    uint4 va[QSA_XVT];
    FUNC_CALL(qsa_xmx_v_load)(value_cache, voff, lv, worker, va);
    FUNC_CALL(qsa_xmx_pv_regs)(va, p, ro);
}
#    endif
#endif  // QSA_ATTENTION_HELPERS
