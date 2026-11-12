// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q1 - PagedQSAIndexer prepare.
//
// One subgroup owns one (sequence, QSA block, role) triple. Roles 0..HIQ-1 normalise + rotate one index-query head
// of every new token of the block; role HIQ writes the raw key cache, the per-block rope row and, once the block is
// complete, the block summary (mean pooling -> f16 -> RMSNorm -> RoPE). The roles are independent, so no SLM and no
// barrier is needed: the pooled rows of this call are read from the input, older rows from the raw key cache.
//
// JIT: QSA_IDX_QUERY_HEADS, QSA_IDX_HEAD_DIM, QSA_IDX_ROTARY_DIM, QSA_COMPRESS_RATIO, QSA_PAGE_SIZE,
//      QSA_IDX_NORM_EPS, SUBGROUP_SIZE
// Dispatch: gws = [SG, HIQ + 1, num_wg], lws = [SG, 1, 1]

#define QSA_INDEXER_HELPERS
#include "qsa_common.cl"

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(qsa_indexer_prepare_paged)(const __global half* q_idx,                  // [tokens, Hiq*Di]
                                  const __global half* k_idx,                  // [tokens, Di]
                                  const __global half* q_norm_weight,          // [Di]
                                  const __global half* k_norm_weight,          // [Di]
                                  const __global half* rope_cos,               // [tokens, rotary_dim]
                                  const __global half* rope_sin,               // [tokens, rotary_dim]
                                  const __global int* past_lens,               // [num_seqs]
                                  const __global int* subsequence_begins,      // [num_seqs+1]
                                  const __global int* block_indices,           // [used_blocks]
                                  const __global int* block_indices_begins,    // [num_seqs+1]
                                  __global half* raw_k_cache,                  // [num_blocks, 1, page_size, Di]
                                  __global half* summary_cache,                // [num_blocks, 1, page_size/r, Di]
                                  __global half* rope_cache,                   // [num_blocks, page_size/r, 2*rotary_dim]
                                  __global half* indexer_q,                    // [tokens, Hiq, Di]
                                  const uint num_wg,
                                  const uint num_seqs) {
    const uint HIQ = QSA_IDX_QUERY_HEADS;
    const uint DI = QSA_IDX_HEAD_DIM;
    const uint ROT = QSA_IDX_ROTARY_DIM;
    const uint R = QSA_COMPRESS_RATIO;
    const uint PAGE = QSA_PAGE_SIZE;
    const uint SPP = PAGE / R;

    const uint wg = get_group_id(2);
    const uint role = get_group_id(1);
    const uint lane = get_sub_group_local_id();
    if (wg >= num_wg) {
        return;
    }

    // Maps the work-group index onto the (sequence, QSA block) it prepares.
    uint seq = 0, qsa_block = 0;
    bool found = false;
    {
        uint acc = 0;
        for (uint i = 0; i < num_seqs && !found; i++) {
            const uint q_len = (uint)subsequence_begins[i + 1] - (uint)subsequence_begins[i];
            if (q_len == 0) {
                continue;
            }
            const uint past = (uint)past_lens[i];
            const uint first = past / R;
            const uint n = (past + q_len - 1) / R - first + 1;
            if (wg < acc + n) {
                seq = i;
                qsa_block = first + (wg - acc);
                found = true;
            }
            acc += n;
        }
    }
    if (!found) {
        return;
    }

    const uint seq_begin = (uint)subsequence_begins[seq];
    const uint q_len = (uint)subsequence_begins[seq + 1] - seq_begin;
    const uint past_len = (uint)past_lens[seq];
    const uint total_len = past_len + q_len;
    const uint page_begin = (uint)block_indices_begins[seq];

    const uint block_start = qsa_block * R;
    const uint block_end = block_start + R;
    const uint lo = block_start > past_len ? block_start : past_len;
    const uint hi = block_end < total_len ? block_end : total_len;

    if (role < HIQ) {
        for (uint p = lo; p < hi; p++) {
            const uint token_idx = seq_begin + (p - past_len);
            float q[QSA_NE];
            FUNC_CALL(qsa_rd_f32)(q_idx + (token_idx * HIQ + role) * DI, q, QSA_NE);
            FUNC_CALL(qsa_rms_norm)(q, q_norm_weight);
#if QSA_IDX_ROTARY_DIM > 0
            FUNC_CALL(qsa_rope)(q, rope_cos + token_idx * ROT, rope_sin + token_idx * ROT, 1);
#endif
            FUNC_CALL(qsa_wr_f32)(indexer_q + (token_idx * HIQ + role) * DI, q, QSA_NE);
        }
        return;
    }

    // Index key: cache layout transform only - no norm/RoPE before the write.
    for (uint p = lo; p < hi; p++) {
        const uint token_idx = seq_begin + (p - past_len);
        const uint page = (uint)block_indices[page_begin + p / PAGE];
        const uint dst = (page * PAGE + p % PAGE) * DI;
        ushort kh[QSA_NE];
        FUNC_CALL(qsa_rd)((const __global ushort*)(k_idx + token_idx * DI), kh, QSA_NE);
        FUNC_CALL(qsa_wr)((__global ushort*)(raw_k_cache + dst), kh, QSA_NE);
#if QSA_IDX_ROTARY_DIM > 0
        if (p == block_start) {
            // One row per block: the only row ever read back for it.
            const uint row = (page * SPP + qsa_block % SPP) * 2 * ROT;
            for (uint i = lane; i < ROT; i += QSA_SG) {
                rope_cache[row + 2 * i] = rope_cos[token_idx * ROT + i];
                rope_cache[row + 2 * i + 1] = rope_sin[token_idx * ROT + i];
            }
        }
#endif
    }

    // Only a fully written block produces a summary.
    if (block_end > total_len) {
        return;
    }

    float acc[QSA_NE];
    unroll_for (uint j = 0; j < QSA_NE; j++) {
        acc[j] = 0.0f;
    }
    for (uint s = 0; s < R; s++) {
        const uint p = block_start + s;
        float v[QSA_NE];
        if (p >= past_len) {
            FUNC_CALL(qsa_rd_f32)(k_idx + (seq_begin + (p - past_len)) * DI, v, QSA_NE);
        } else {
            const uint page = (uint)block_indices[page_begin + p / PAGE];
            FUNC_CALL(qsa_rd_f32)(raw_k_cache + (page * PAGE + p % PAGE) * DI, v, QSA_NE);
        }
        unroll_for (uint j = 0; j < QSA_NE; j++) {
            acc[j] += v[j];
        }
    }
    // The pooled mean is rounded to f16 before RMSNorm, like the reference model.
    unroll_for (uint j = 0; j < QSA_NE; j++) {
        acc[j] = convert_float(convert_half_rte(acc[j] * (1.0f / (float)R)));
    }
    FUNC_CALL(qsa_rms_norm)(acc, k_norm_weight);
    const uint page = (uint)block_indices[page_begin + qsa_block / SPP];
#if QSA_IDX_ROTARY_DIM > 0
    // Coefficients of the block's first token: this call's input row, else the row persisted earlier.
    if (block_start >= past_len) {
        const uint first_token = seq_begin + (block_start - past_len);
        FUNC_CALL(qsa_rope)(acc, rope_cos + first_token * ROT, rope_sin + first_token * ROT, 1);
    } else {
        const uint row = (page * SPP + qsa_block % SPP) * 2 * ROT;
        FUNC_CALL(qsa_rope)(acc, rope_cache + row, rope_cache + row + 1, 2);
    }
#endif
    FUNC_CALL(qsa_wr_f32)(summary_cache + (page * SPP + qsa_block % SPP) * DI, acc, QSA_NE);
}
