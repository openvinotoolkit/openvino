// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q3 decode, split-selected-KV variant, pass 1.
//
// Decode has one query per sequence, so a query-owned subgroup would leave the GPU idle. Each token's own selected
// list is split over QSA_SPLIT_PARTITIONS independent subgroups, each producing a partial (max, sum, weighted-V)
// state; sparse_attention_split_reduce.cl merges them. The last partition also owns the causal tail (also for K = 0).
// Every partition writes its whole row (zeros when empty): the reduce pass reads all of them unconditionally.
//
// JIT: as sparse_attention_scalar.cl plus QSA_SPLIT_PARTITIONS
// Dispatch: gws = [QSA_HEADS * SG, num_tokens (paged) | B*L (dense), QSA_SPLIT_PARTITIONS], lws = [SG, 1, 1]

#define QSA_ATTENTION_HELPERS
#include "qsa_common.cl"

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_split)(QSA_ATT_INPUT_ARGS,
                               __global float* partial_acc,  // [tokens, Hq, P, Dv]
                               __global float* partial_ml,   // [tokens, Hq, P, 2]
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
    const uint P = QSA_SPLIT_PARTITIONS;
    const uint lane = get_sub_group_local_id();

    const uint head = get_group_id(0);
    const uint part = get_group_id(2);
    const uint kv_head = head / (HQ / HKV);

#if QSA_ADDRESS_MODE == 0
    const uint token_idx = get_group_id(1);
    if (head >= HQ || token_idx >= num_tokens || part >= P) {
        return;
    }
    uint seq = 0, abs_pos = 0;
    if (!FUNC_CALL(qsa_resolve_token_paged)(subsequence_begins, past_lens, num_seqs, token_idx, &seq, &abs_pos)) {
        return;
    }
    const uint page_begin = (uint)block_indices_begins[seq];
    const uint q_elem = q_base + token_idx * q_pitch + head * DH;
    const uint sel_row = token_idx;
    const uint count = (uint)selected_counts[sel_row];
    const bool has_tail = true;
#else
    const uint row = get_group_id(1);
    if (head >= HQ || row >= num_rows || part >= P) {
        return;
    }
    const uint batch = row / seq_len;
    const uint l = row - batch * seq_len;
    uint kl = (uint)key_length[batch];
    kl = kl < history_len ? kl : history_len;
    const uint shift = history_len - kl;
    const bool pad_row = kl + l < seq_len;
    const uint abs_pos = kl + l - seq_len;
    const uint q_elem = ((batch * HQ + head) * seq_len + l) * DH;
    const uint sel_row = row;
    const uint count = pad_row ? 0 : (uint)selected_counts[sel_row];
    const bool has_tail = !pad_row;
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

    // Split slot ranges, not absolute positions, so every partition's share of the list is contiguous.
    const uint selected = count < K_TOPK ? count : K_TOPK;
    const uint begin = selected * part / P;
    const uint end = selected * (part + 1) / P;
    for (uint i = begin; i < end; i++) {
        const int b = selected_blocks[sel_row * K_TOPK + i];
        if (b < 0) {
            continue;  // must precede any KV address computation
        }
        const uint pos0 = (uint)b * R;
        const uint kbase = QSA_KROW(pos0);
        const uint vbase = QSA_VROW(pos0);
        unroll_for (uint s = 0; s < R; s++) {
            FUNC_CALL(qsa_scalar_attend)(key_cache + kbase + s * DH, value_cache + vbase + s * DV, q, acc, &running_max, &running_sum);
        }
    }
    if (part == P - 1 && has_tail) {
        for (uint pos = FUNC_CALL(qsa_num_complete_blocks)(abs_pos) * R; pos <= abs_pos; pos++) {
            FUNC_CALL(qsa_scalar_attend)(key_cache + QSA_KROW(pos), value_cache + QSA_VROW(pos), q, acc, &running_max, &running_sum);
        }
    }

#if QSA_ADDRESS_MODE == 0
    const uint prow = (token_idx * HQ + head) * P + part;
#else
    const uint prow = (row * HQ + head) * P + part;
#endif
    if (lane == 0) {
        partial_ml[prow * 2] = running_max;
        partial_ml[prow * 2 + 1] = running_sum;
    }
    unroll_for (uint j = 0; j < QSA_NEV; j++) {
        partial_acc[prow * DV + lane + QSA_SG * j] = acc[j];
    }
}
