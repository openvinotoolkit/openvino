// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q0 - SparsePA main K/V cache write. Two kernels share this file, selected by QSA_KV_WRITE_KCHANNEL.
//
// Default: one subgroup copies (f16 caches) or quantizes (BY_TOKEN quantized caches) the K and V row of one
// (token, kv head) into the paged cache. The input rows may be padded and offset (key_pitch / key_offset), so they are
// read lane-wise without any alignment assumption. BY_CHANNEL keys are skipped here.
// JIT: QSA_KV_HEADS, QSA_K_HEAD_SIZE, QSA_V_HEAD_SIZE, QSA_PAGE_SIZE, SUBGROUP_SIZE,
//      QSA_{K,V}_{MODE,BITS,SYM,GROUP,ROW_BYTES,PAGE_BYTES,PARAM_OFFSET} (see qsa_common.cl)
// Dispatch: gws = [SG, QSA_KV_HEADS, tokens_padded], lws = [SG, 1, 1]
//
// QSA_KV_WRITE_KCHANNEL: BY_CHANNEL key cache write. One subgroup owns one (sequence, token group, kv head); a group is
// QSA_K_GROUP consecutive tokens of one page that share one scale / zero point per channel. A group that the new tokens
// complete is re-quantized exactly: pass 1 gets the per-channel range (cached tokens dequantized with their old
// parameters, new tokens from the key input), pass 2 re-quantizes and rewrites every row, then the parameters.
// A group that stays partial (QSA_K_INCREMENTAL, decode) keeps its parameters when the new rows fit the stored range, so
// only the new rows are written. Otherwise the range becomes (stored range or, for the first token of a group, the range
// of the previous group) united with the new rows widened by QSA_K_HEADROOM, and the group is rewritten once.
// Groups without a new token are never touched.
// Dispatch: gws = [(tokens / QSA_K_GROUP + 2 * num_seqs) * SG, QSA_KV_HEADS, 1], lws = [SG, 1, 1]

#define QSA_KVQ_HELPERS
#include "qsa_common.cl"

#if QSA_KV_WRITE_KCHANNEL

#define KCH_NE (QSA_K_HEAD_SIZE / QSA_SG)

#if (QSA_K_HEAD_SIZE % SUBGROUP_SIZE) != 0 || QSA_K_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE
#    error "head size must be a whole number of subgroup rows of at most QSA_MAX_NE elements per lane"
#endif
#if QSA_K_MODE != 2
#    error "this kernel only writes BY_CHANNEL keys"
#endif
#if (QSA_PAGE_SIZE % QSA_K_GROUP) != 0
#    error "a group must fit a page"
#endif

// Maps work-group `wg` of a grid sized `tokens / group + 2 * num_seqs` onto the (sequence, token group) it re-quantizes:
// sequence i owns the groups touched by its new tokens.
inline bool FUNC(qsa_resolve_group)(const __global int* subsequence_begins,
                                    const __global int* past_lens,
                                    const uint num_seqs,
                                    const uint wg,
                                    const uint group_size,
                                    __private uint* seq,
                                    __private uint* group) {
    uint acc = 0;
    for (uint i = 0; i < num_seqs; i++) {
        const uint q_len = (uint)subsequence_begins[i + 1] - (uint)subsequence_begins[i];
        if (q_len == 0) {
            continue;
        }
        const uint past = (uint)past_lens[i];
        const uint first = past / group_size;
        const uint n = (past + q_len - 1) / group_size - first + 1;
        if (wg < acc + n) {
            *seq = i;
            *group = first + (wg - acc);
            return true;
        }
        acc += n;
    }
    return false;
}

// Value range [lo, hi] a channel with parameters (scale, zp) can represent.
#define QSA_KCH_NONE 3.0e38f  // "no range yet" bound of an asymmetric range
#define QSA_KCH_ZR (QSA_K_SYM || QSA_K_ZERO_RANGE)
// Smallest asymmetric range relative to the larger magnitude: keeps |zero point| <= 512, where (code - zp) is still accurate in f16.
#define QSA_KCH_MINW ((QSA_K_BITS == 8 ? 255.0f : 15.0f) / 512.0f)
inline float2 FUNC(qsa_kch_range)(const float scale, const float zp) {
#    if QSA_K_SYM
    const float hi = (QSA_K_BITS == 8 ? 127.0f : 7.0f) * scale;
    return (float2)(-hi, hi);
#    else
    return (float2)(-zp * scale, ((QSA_K_BITS == 8 ? 255.0f : 15.0f) - zp) * scale);
#    endif
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_kv_write_kchannel)(const __global half* key,                  // [num_tokens, Hkv*Dh] token-major
                                           const __global int* past_lens,             // [num_seqs]
                                           const __global int* block_indices,         // [used_blocks]
                                           const __global int* block_indices_begins,  // [num_seqs+1]
                                           const __global int* subsequence_begins,    // [num_seqs+1]
                                           __global uchar* key_cache,                 // [num_blocks, Hkv, page_size + parameter rows, row bytes]
                                           const uint key_pitch,
                                           const uint key_offset,
                                           const uint num_seqs) {
    const uint KV_HEADS = QSA_KV_HEADS;
    const uint DH = QSA_K_HEAD_SIZE;
    const uint PAGE = QSA_PAGE_SIZE;
    const uint G = QSA_K_GROUP;
    const uint BITS = QSA_K_BITS;
    const uint ROWB = QSA_K_ROW_BYTES;
    const uint SYM = QSA_K_SYM;
    const uint REC_BYTES = DH * (SYM ? 2 : 4);
    const float QMAX = BITS == 8 ? 255.0f : 15.0f;

    const uint head = get_group_id(1);
    const uint lane = get_sub_group_local_id();
    uint seq = 0, group = 0;
    if (head >= KV_HEADS || !FUNC_CALL(qsa_resolve_group)(subsequence_begins, past_lens, num_seqs, get_group_id(0), G, &seq, &group)) {
        return;
    }

    const uint seq_begin = (uint)subsequence_begins[seq];
    const uint past = (uint)past_lens[seq];
    const uint total = past + (uint)subsequence_begins[seq + 1] - seq_begin;
    const uint g0 = group * G;
    const uint n_valid = total - g0 < G ? total - g0 : G;
    const uint n_old = past > g0 ? (past - g0 < n_valid ? past - g0 : n_valid) : 0;

    const uint tok0 = g0 % PAGE;
    const uint physical_page = (uint)block_indices[(uint)block_indices_begins[seq] + g0 / PAGE];
    const uint pb = (physical_page * KV_HEADS + head) * QSA_K_PAGE_BYTES;
    const uint rec = pb + QSA_K_PARAM_OFFSET + (tok0 / G) * REC_BYTES;

    float old_scale[QSA_K_HEAD_SIZE / QSA_SG];
    float inv[QSA_K_HEAD_SIZE / QSA_SG];  // 1 / scale of the stored parameters until the group is re-quantized
    float old_zp[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < KCH_NE; j++) {
        const uint c = lane + QSA_SG * j;
        old_scale[j] = 1.0f;
        old_zp[j] = 0.0f;
        if (n_old > 0) {
            old_scale[j] = convert_float(*(const __global half*)(key_cache + rec + c * 2));
            old_zp[j] = SYM ? FUNC_CALL(qsa_q_zero_sym)(BITS) : convert_float(*(const __global half*)(key_cache + rec + DH * 2 + c * 2));
        }
        inv[j] = 1.0f / old_scale[j];
    }

    // Row t of the group as f32 (cached rows dequantized with their old parameters in f16 arithmetic).
#define KCH_LOAD_ROW(t, h)                                                                                                          \
    {                                                                                                                               \
        if ((t) < n_old) {                                                                                                          \
            const uint row_off = pb + (tok0 + (t)) * ROWB;                                                                          \
            unroll_for (uint j = 0; j < KCH_NE; j++) {                                                                                  \
                const uint c = lane + QSA_SG * j;                                                                                   \
                float code;                                                                                                         \
                if (BITS == 8) {                                                                                                    \
                    code = convert_float(key_cache[row_off + c]);                                                                   \
                } else {                                                                                                            \
                    const uchar b = key_cache[row_off + (c >> 1)];                                                                  \
                    code = convert_float((c & 1) ? (b >> 4) : (b & 0xF));                                                           \
                }                                                                                                                   \
                h[j] = FUNC_CALL(qsa_q_dq)(code, old_zp[j], old_scale[j]);                                          \
            }                                                                                                                       \
        } else {                                                                                                                    \
            const __global ushort* src = (const __global ushort*)key + key_offset + (seq_begin + (g0 + (t) - past)) * key_pitch + head * DH + lane; \
            unroll_for (uint j = 0; j < KCH_NE; j++) {                                                                                  \
                h[j] = convert_float(as_half(src[j * QSA_SG]));                                                                     \
            }                                                                                                                       \
        }                                                                                                                           \
    }

    // Pass 1: per-channel range. Symmetric ranges include 0, asymmetric ones are the data range (see the guard below).
    float mn[QSA_K_HEAD_SIZE / QSA_SG];
    float mx[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < KCH_NE; j++) {
        mn[j] = QSA_KCH_ZR ? 0.0f : QSA_KCH_NONE;
        mx[j] = QSA_KCH_ZR ? 0.0f : -QSA_KCH_NONE;
    }
    // A group that stays partial only reads its new rows here; the cached ones are covered by the stored range.
    const bool incremental = QSA_K_INCREMENTAL && n_valid < G;
    bool fit = incremental && n_old > 0;
    for (uint t = incremental ? n_old : 0; t < n_valid; t++) {
        float h[QSA_K_HEAD_SIZE / QSA_SG];
        KCH_LOAD_ROW(t, h);
        unroll_for (uint j = 0; j < KCH_NE; j++) {
            mn[j] = fmin(mn[j], h[j]);
            mx[j] = fmax(mx[j], h[j]);
            // A code in [-0.5, QMAX + 0.5) rounds with the usual half-step error.
            const float v = h[j] * inv[j] + old_zp[j];
            fit = fit && v >= -0.5f && v < QMAX + 0.5f;
        }
    }
    fit = sub_group_all(fit);
    if (incremental && !fit) {
        // Stored range, or for the first token of a group the previous group's one, united with the new rows plus headroom.
        const bool from_prev = n_old == 0 && g0 >= G && past == g0;  // else a work-group of this launch may rewrite it
        uint prev_rec = 0;
        if (from_prev) {
            const uint prev = g0 - G;
            const uint prev_page = (uint)block_indices[(uint)block_indices_begins[seq] + prev / PAGE];
            prev_rec = (prev_page * KV_HEADS + head) * QSA_K_PAGE_BYTES + QSA_K_PARAM_OFFSET + ((prev % PAGE) / G) * REC_BYTES;
        }
        unroll_for (uint j = 0; j < KCH_NE; j++) {
            const uint c = lane + QSA_SG * j;
            float2 range = (float2)(QSA_KCH_ZR ? 0.0f : QSA_KCH_NONE, QSA_KCH_ZR ? 0.0f : -QSA_KCH_NONE);
            if (n_old > 0) {
                range = FUNC_CALL(qsa_kch_range)(old_scale[j], old_zp[j]);
            } else if (from_prev) {
                const float s = convert_float(*(const __global half*)(key_cache + prev_rec + c * 2));
                const float z = SYM ? FUNC_CALL(qsa_q_zero_sym)(BITS) : convert_float(*(const __global half*)(key_cache + prev_rec + DH * 2 + c * 2));
                range = FUNC_CALL(qsa_kch_range)(s, z);
            }
            if (QSA_KCH_ZR) {
                mn[j] = fmin(range.s0, mn[j] * (1.0f + QSA_K_HEADROOM));
                mx[j] = fmax(range.s1, mx[j] * (1.0f + QSA_K_HEADROOM));
            } else {
                const float w = fmax(mx[j] - mn[j], QSA_KCH_MINW * fmax(fabs(mn[j]), fabs(mx[j])));
                mn[j] = fmin(range.s0, mn[j] - QSA_K_HEADROOM * w);
                mx[j] = fmax(range.s1, mx[j] + QSA_K_HEADROOM * w);
            }
        }
    }

    // New parameters (the old ones stay in registers for dequantizing the cached rows below).
    float zero_f[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < KCH_NE; j++) {
        zero_f[j] = old_zp[j];
        if (!fit) {
            const uint c = lane + QSA_SG * j;
            float scale, zero;
            if (SYM) {
                scale = fmax(fmax(mx[j], -mn[j]) / (BITS == 8 ? 127.0f : 7.0f), QSA_KVQ_MIN_SCALE);
                zero = FUNC_CALL(qsa_q_zero_sym)(BITS);
            } else if (QSA_K_ZERO_RANGE) {
                scale = fmax((mx[j] - mn[j]) / QMAX, QSA_KVQ_MIN_SCALE);
                zero = -mn[j] / scale;
            } else {
                // Widen a narrow range upwards to QSA_KCH_MINW of the larger magnitude (a centered value would sit on a rounding tie).
                const float widen = fmax(QSA_KCH_MINW * fmax(fabs(mn[j]), fabs(mx[j])) - (mx[j] - mn[j]), 0.0f);
                scale = fmax((mx[j] + widen - mn[j]) / QMAX, QSA_KVQ_MIN_SCALE);
                zero = -mn[j] / scale;
            }
            const half scale_h = convert_half_rte(scale);
            const half zero_h = convert_half_rte(zero);
            inv[j] = 1.0f / convert_float(scale_h);
            zero_f[j] = convert_float(zero_h);
            *(__global half*)(key_cache + rec + c * 2) = scale_h;
            if (!SYM) {
                *(__global half*)(key_cache + rec + DH * 2 + c * 2) = zero_h;
            }
        }
    }

    // Pass 2: quantize the new rows when the stored parameters are kept, else re-quantize the whole group.
    for (uint t = fit ? n_old : 0; t < n_valid; t++) {
        float h[QSA_K_HEAD_SIZE / QSA_SG];
        KCH_LOAD_ROW(t, h);
        uint code[QSA_K_HEAD_SIZE / QSA_SG];
        unroll_for (uint j = 0; j < KCH_NE; j++) {
            code[j] = (uint)(clamp(h[j] * inv[j] + zero_f[j], 0.0f, QMAX) + 0.5f);
        }
        FUNC_CALL(qsa_q_store_codes)(key_cache, pb + (tok0 + t) * ROWB, code, KCH_NE, BITS);
    }
}

#else  // QSA_KV_WRITE_KCHANNEL

#if (QSA_K_HEAD_SIZE % SUBGROUP_SIZE) != 0 || (QSA_V_HEAD_SIZE % SUBGROUP_SIZE) != 0 || QSA_K_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE || QSA_V_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE
#    error "head sizes must be whole subgroup rows of at most QSA_MAX_NE elements per lane"
#endif

#if QSA_K_MODE != 0
#    define QSA_KV_K_T uchar
#else
#    define QSA_KV_K_T half
#endif
#if QSA_V_MODE != 0
#    define QSA_KV_V_T uchar
#else
#    define QSA_KV_V_T half
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_kv_write)(const __global half* key,                   // [num_tokens, Hkv*Dh] token-major
                                  const __global half* value,                 // [num_tokens, Hkv*Dv] token-major
                                  const __global int* past_lens,              // [num_seqs]
                                  const __global int* block_indices,          // [used_blocks]
                                  const __global int* block_indices_begins,   // [num_seqs+1]
                                  const __global int* subsequence_begins,     // [num_seqs+1]
                                  __global QSA_KV_K_T* key_cache,             // [num_blocks, Hkv, page_size (+ parameter rows), row]
                                  __global QSA_KV_V_T* value_cache,           // [num_blocks, Hkv, page_size (+ parameter rows), row]
                                  const uint key_pitch,
                                  const uint key_offset,
                                  const uint value_pitch,
                                  const uint value_offset,
                                  const uint num_seqs) {
    const uint KV_HEADS = QSA_KV_HEADS;
    const uint K_HEAD = QSA_K_HEAD_SIZE;
    const uint V_HEAD = QSA_V_HEAD_SIZE;
    const uint PAGE = QSA_PAGE_SIZE;

    const uint head_idx = get_group_id(1);
    const uint token_idx = get_group_id(2);
    const uint lane = get_sub_group_local_id();

    if (token_idx >= (uint)subsequence_begins[num_seqs]) {
        return;
    }

    uint seq = 0;
    for (uint i = 0; i < num_seqs; i++) {
        if (token_idx >= (uint)subsequence_begins[i] && token_idx < (uint)subsequence_begins[i + 1]) {
            seq = i;
            break;
        }
    }

    const uint abs_pos = (uint)past_lens[seq] + token_idx - (uint)subsequence_begins[seq];
    const uint physical_page = (uint)block_indices[(uint)block_indices_begins[seq] + abs_pos / PAGE];
    const uint token_in_page = abs_pos % PAGE;

    const uint k_src = key_offset + token_idx * key_pitch + head_idx * K_HEAD;
    const uint v_src = value_offset + token_idx * value_pitch + head_idx * V_HEAD;
    const uint kv_head = head_idx;
    const __global ushort* ks = (const __global ushort*)key + k_src + lane;
    const __global ushort* vs = (const __global ushort*)value + v_src + lane;

    // Every load is issued before the first store: the source and cache pointers may alias as far as the compiler
    // knows, so interleaving them would serialize one memory round trip per element.
#if QSA_K_MODE == 0
    ushort kr[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < K_HEAD / QSA_SG; j++) {
        kr[j] = ks[j * QSA_SG];
    }
#elif QSA_K_MODE == 1
    float kf[QSA_K_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < K_HEAD / QSA_SG; j++) {
        kf[j] = convert_float(as_half(ks[j * QSA_SG]));
    }
#endif
#if QSA_V_MODE == 0
    ushort vr[QSA_V_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < V_HEAD / QSA_SG; j++) {
        vr[j] = vs[j * QSA_SG];
    }
#else
    float vf[QSA_V_HEAD_SIZE / QSA_SG];
    unroll_for (uint j = 0; j < V_HEAD / QSA_SG; j++) {
        vf[j] = convert_float(as_half(vs[j * QSA_SG]));
    }
#endif

#if QSA_K_MODE == 0
    FUNC_CALL(qsa_wr)((__global ushort*)(key_cache + FUNC_CALL(qsa_kv_cache_offset)(physical_page, head_idx, token_in_page, KV_HEADS, PAGE, K_HEAD)), kr, K_HEAD / QSA_SG);
#elif QSA_K_MODE == 1
    {
        uint ro, po;
        QSA_K_LOC(physical_page, token_in_page, &ro, &po);
        uint code[QSA_K_HEAD_SIZE / QSA_SG];
        half sh, zh;
        FUNC_CALL(qsa_q_quant_row)(kf, code, K_HEAD / QSA_SG, QSA_K_BITS, QSA_K_SYM, &sh, &zh);
        FUNC_CALL(qsa_q_store_codes)((__global uchar*)key_cache, ro, code, K_HEAD / QSA_SG, QSA_K_BITS);
        if (lane == 0) {
            *(__global uint*)((__global uchar*)key_cache + po) = (uint)as_ushort(sh) | ((uint)as_ushort(zh) << 16);
        }
    }
#endif

#if QSA_V_MODE == 0
    FUNC_CALL(qsa_wr)((__global ushort*)(value_cache + FUNC_CALL(qsa_kv_cache_offset)(physical_page, head_idx, token_in_page, KV_HEADS, PAGE, V_HEAD)), vr, V_HEAD / QSA_SG);
#else
    {
        uint ro, po;
        QSA_V_LOC(physical_page, token_in_page, &ro, &po);
        uint code[QSA_V_HEAD_SIZE / QSA_SG];
        half sh, zh;
        FUNC_CALL(qsa_q_quant_row)(vf, code, V_HEAD / QSA_SG, QSA_V_BITS, QSA_V_SYM, &sh, &zh);
        FUNC_CALL(qsa_q_store_codes)((__global uchar*)value_cache, ro, code, V_HEAD / QSA_SG, QSA_V_BITS);
        if (lane == 0) {
            *(__global uint*)((__global uchar*)value_cache + po) = (uint)as_ushort(sh) | ((uint)as_ushort(zh) << 16);
        }
    }
#endif
}

#endif  // QSA_KV_WRITE_KCHANNEL
