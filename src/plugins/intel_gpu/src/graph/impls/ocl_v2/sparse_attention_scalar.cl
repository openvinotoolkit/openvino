// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q3 - selected-block sparse attention, scalar variant, shared by
// SparsePA (QSA_ADDRESS_MODE 0, paged K/V) and SparseSDPA (QSA_ADDRESS_MODE 1, dense left-padded K/V).
//
// One subgroup = one (query head, token); lanes cover the head dimensions, a dot product is a per-lane partial sum
// plus a subgroup reduction. Every selected block (r tokens) is followed by the always-visible causal tail, with an
// online softmax over all of them.
//
// JIT: QSA_HEADS, QSA_KV_HEADS, QSA_K_HEAD_SIZE, QSA_V_HEAD_SIZE, QSA_COMPRESS_RATIO, QSA_BLOCK_TOPK, QSA_SCALE,
//      QSA_ADDRESS_MODE, SUBGROUP_SIZE and (paged) QSA_PAGE_SIZE
// Dispatch: gws = [QSA_HEADS * SG, num_tokens (paged) | B*L (dense), 1], lws = [SG, 1, 1]

#define QSA_ATTENTION_HELPERS
#include "qsa_common.cl"

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_scalar)(QSA_ATT_INPUT_ARGS,
                                __global half* output,  // paged: [num_tokens, Hq*Dv]; dense: [B, Hq, L, Dv]
#if QSA_ADDRESS_MODE == 0
                                const uint num_tokens,
                                const uint num_seqs,
                                const uint q_base,
                                const uint q_pitch
#else
                                const uint seq_len,
                                const uint history_len,
                                const uint num_rows,
                                const uint k_pitch,
                                const uint v_pitch
#endif
) {
    const uint HQ = QSA_HEADS;
    const uint HKV = QSA_KV_HEADS;
    const uint DH = QSA_K_HEAD_SIZE;
    const uint DV = QSA_V_HEAD_SIZE;
    const uint R = QSA_COMPRESS_RATIO;
    const uint K_TOPK = QSA_BLOCK_TOPK;
    const uint lane = get_sub_group_local_id();

    const uint head = get_group_id(0);
    const uint kv_head = head / (HQ / HKV);

#if QSA_ADDRESS_MODE == 0
    const uint token_idx = get_group_id(1);
    if (head >= HQ || token_idx >= num_tokens) {
        return;
    }
    uint seq = 0, abs_pos = 0;
    if (!FUNC_CALL(qsa_resolve_token_paged)(subsequence_begins, past_lens, num_seqs, token_idx, &seq, &abs_pos)) {
        return;
    }
    const uint page_begin = (uint)block_indices_begins[seq];
    const uint q_elem = q_base + token_idx * q_pitch + head * DH;
    const uint sel_row = token_idx;
    const uint selected = (uint)selected_counts[sel_row];
    const bool has_tail = true;
    const uint out_elem = (token_idx * HQ + head) * DV;
#else
    const uint row = get_group_id(1);
    if (head >= HQ || row >= num_rows) {
        return;
    }
    const uint batch = row / seq_len;
    const uint l = row - batch * seq_len;
    uint kl = (uint)key_length[batch];
    kl = kl < history_len ? kl : history_len;
    // Left-padded K/V: logical key j of this row is stored at row (history_len - kl) + j.
    const uint shift = history_len - kl;
    const bool pad_row = kl + l < seq_len;
    const uint abs_pos = kl + l - seq_len;
    const uint q_elem = ((batch * HQ + head) * seq_len + l) * DH;
    const uint sel_row = row;
    const uint selected = pad_row ? 0 : (uint)selected_counts[sel_row];
    const bool has_tail = !pad_row;
    const uint out_elem = ((batch * HQ + head) * seq_len + l) * DV;
#endif

    float q[QSA_NEK];
    unroll_for (uint j = 0; j < QSA_NEK; j++) {
        q[j] = convert_float(query[q_elem + lane + QSA_SG * j]) * (float)QSA_SCALE;
    }

    float running_max = -1e30f;
    float running_sum = 0.0f;
    float acc[QSA_NEV];
    unroll_for (uint j = 0; j < QSA_NEV; j++) {
        acc[j] = 0.0f;
    }

    for (uint i = 0; i < selected && i < K_TOPK; i++) {
        const int b = selected_blocks[sel_row * K_TOPK + i];
        if (b < 0) {
            continue;
        }
        const uint pos0 = (uint)b * R;
        const uint kbase = QSA_KROW(pos0);
        const uint vbase = QSA_VROW(pos0);
        unroll_for (uint s = 0; s < R; s++) {
            FUNC_CALL(qsa_scalar_attend)(key_cache + kbase + s * DH, value_cache + vbase + s * DV, q, acc, &running_max, &running_sum);
        }
    }

    if (has_tail) {
        const uint tail_begin = FUNC_CALL(qsa_num_complete_blocks)(abs_pos) * R;
        for (uint p = tail_begin; p <= abs_pos; p++) {
            FUNC_CALL(qsa_scalar_attend)(key_cache + QSA_KROW(p), value_cache + QSA_VROW(p), q, acc, &running_max, &running_sum);
        }
    }

    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    float o[QSA_NEV];
    unroll_for (uint j = 0; j < QSA_NEV; j++) {
        o[j] = acc[j] * inv_sum;
    }
    FUNC_CALL(qsa_wr_f32)(output + out_elem, o, QSA_NEV);
}
