// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q2 score, decode-partition variant, shared by the paged and
// dense indexers through QSA_ADDRESS_MODE (0 = paged page-table gather, 1 = dense batch/stride addressing).
//
// One subgroup scores one partition of QSA_SCORE_PARTITION_BLOCKS blocks of one query token. Lanes cover the index
// head dimension (block reads), so each block/head dot product is a per-lane partial sum plus a subgroup reduction;
// the query stays in registers and U summary rows are kept in flight. It is also the scorer of the non-XMX prefill.
//
// JIT: QSA_IDX_QUERY_HEADS, QSA_IDX_HEAD_DIM, QSA_IDX_SCALE, QSA_COMPRESS_RATIO, QSA_ADDRESS_MODE, QSA_BLOCK_TOPK,
//      QSA_SCORE_PARTITION_BLOCKS, QSA_DENSE_SCORE_BYPASS, SUBGROUP_SIZE and (paged) QSA_PAGE_SIZE
// Dispatch: gws = [(row_end - row_begin) * SG, 1, num_partitions], lws = [SG, 1, 1]
// `pitch` is the (8-aligned) row pitch of block_scores, which holds only rows [row_begin, row_end).

#define QSA_INDEXER_HELPERS
#include "qsa_common.cl"

#define QSA_U 8

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(qsa_indexer_score_partition)(const __global half* indexer_q,           // [tokens, Hiq, Di]
                                    const __global half* summary_cache,       // paged: [N_phys,1,page/r,Di]; dense: [B,1,S_cap/r,Di]
                                    const __global int* past_lens,            // paged: [num_seqs]; dense: [B]
#if QSA_ADDRESS_MODE == 0
                                    const __global int* subsequence_begins,   // [num_seqs+1]
                                    const __global int* block_indices,        // [used_blocks]
                                    const __global int* block_indices_begins, // [num_seqs+1]
#endif
                                    __global float* block_scores,             // [row_end - row_begin, pitch]
                                    const uint num_tokens,
#if QSA_ADDRESS_MODE == 0
                                    const uint num_seqs,
#else
                                    const uint seq_len,
                                    const uint summary_stride,
#endif
                                    const uint pitch,
                                    const uint num_partitions,
                                    const uint row_begin,
                                    const uint row_end) {
    const uint HIQ = QSA_IDX_QUERY_HEADS;
    const uint DI = QSA_IDX_HEAD_DIM;
#if QSA_ADDRESS_MODE == 0
    const uint PAGE = QSA_PAGE_SIZE;
    const uint SPP = PAGE / QSA_COMPRESS_RATIO;
#endif
    const uint lane = get_sub_group_local_id();

    const uint token_idx = get_group_id(0) + row_begin;
    const uint partition = get_group_id(2);
    if (token_idx >= row_end || token_idx >= num_tokens || partition >= num_partitions) {
        return;
    }

    uint seq = 0, abs_pos = 0;
#if QSA_ADDRESS_MODE == 0
    if (!FUNC_CALL(qsa_resolve_token_paged)(subsequence_begins, past_lens, num_seqs, token_idx, &seq, &abs_pos)) {
        return;
    }
    const uint page_begin = (uint)block_indices_begins[seq];
#else
    FUNC_CALL(qsa_resolve_token_dense)(past_lens, seq_len, token_idx, &seq, &abs_pos);
#endif

    const uint n_complete = FUNC_CALL(qsa_num_complete_blocks)(abs_pos);
#if QSA_DENSE_SCORE_BYPASS
    // The paired finalizer never reads the scores of a row that selects every block.
    if (n_complete <= (uint)QSA_BLOCK_TOPK) {
        return;
    }
#endif

    const uint begin = partition * (uint)QSA_SCORE_PARTITION_BLOCKS;
    if (begin >= n_complete) {
        return;
    }
    uint end = begin + (uint)QSA_SCORE_PARTITION_BLOCKS;
    if (end > n_complete) {
        end = n_complete;
    }

    __global float* row = block_scores + (size_t)(token_idx - row_begin) * pitch;

    // The query stays in registers for the whole partition scan; only summaries stream.
    float q[QSA_IDX_QUERY_HEADS][QSA_NE];
    unroll_for (uint h = 0; h < HIQ; h++) {
        FUNC_CALL(qsa_rd_f32)(indexer_q + (token_idx * HIQ + h) * DI, q[h], QSA_NE);
    }

    for (uint b0 = begin; b0 < end; b0 += QSA_U) {
        const uint n = (end - b0) < QSA_U ? (end - b0) : QSA_U;
        ushort s[QSA_U][QSA_NE];
        unroll_for (uint u = 0; u < QSA_U; u++) {
            if (u < n) {
                const uint b = b0 + u;
#if QSA_ADDRESS_MODE == 0
                const uint page = (uint)block_indices[page_begin + b / SPP];
                const uint src = (page * SPP + b % SPP) * DI;
#else
                const uint src = (seq * summary_stride + b) * DI;
#endif
                FUNC_CALL(qsa_rd)((const __global ushort*)(summary_cache + src), s[u], QSA_NE);
            }
        }
        float mine = 0.0f;
        unroll_for (uint u = 0; u < QSA_U; u++) {
            if (u < n) {
                float total = 0.0f;
                unroll_for (uint h = 0; h < HIQ; h++) {
                    float dot = 0.0f;
                    unroll_for (uint j = 0; j < QSA_NE; j++) {
                        dot += q[h][j] * convert_float(as_half(s[u][j]));
                    }
                    dot = sub_group_reduce_add(dot);
                    total += dot > 0.0f ? dot : 0.0f;
                }
                if (lane == u) {
                    mine = total * (float)QSA_IDX_SCALE;
                }
            }
        }
        if (lane < n) {
            row[b0 + lane] = mine;
        }
    }
}
