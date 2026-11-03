// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q2 prefill, systolic variant, shared by the paged and dense indexers
// through QSA_ADDRESS_MODE (0 = paged page-table gather, 1 = dense batch/stride addressing).
//
// XMX mapping (subgroup of 16): lane = query. A operand = 8 summary rows (a[i] = S[b0 + i][16 k + lane]), B operand
// = the lane's own 16 query halves of one K tile (kept in SLM, lane contiguous), C = 8 block scores of that query
// (c[i] = score[query = lane][b0 + i]), so the result needs no transpose. Each index head is ReLU-ed before the head
// sum; the sum is scaled by QSA_IDX_SCALE.
//
// JIT: QSA_IDX_QUERY_HEADS, QSA_IDX_HEAD_DIM, QSA_IDX_SCALE, QSA_COMPRESS_RATIO, QSA_BLOCK_TOPK, QSA_ADDRESS_MODE,
//      QSA_SCORE_WG, SUBGROUP_SIZE and (paged) QSA_PAGE_SIZE
// Dispatch: gws = [num_tiles * QSA_SCORE_WG * SG, 1, 1], lws = [QSA_SCORE_WG * SG, 1, 1]; one work-group owns one
// 16-query tile. `pitch` is the row pitch of block_scores, a multiple of 8 so the 8-wide stores stay inside the row.

#define QSA_INDEXER_HELPERS
#include "qsa_common.cl"

#define QSA_KT (QSA_IDX_HEAD_DIM / 16)
#define QSA_MB 8

#if SUBGROUP_SIZE != 16
#    error "the systolic indexer score needs a 16-wide subgroup"
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(qsa_indexer_score_dpas)(const __global half* indexer_q,           // [tokens, Hiq, Di]
                               const __global half* summary_cache,       // paged: [N_phys,1,page/r,Di]; dense: [B,1,S_cap/r,Di]
                               const __global int* past_lens,            // paged: [num_seqs]; dense: [B]
#if QSA_ADDRESS_MODE == 0
                               const __global int* subsequence_begins,   // [num_seqs+1]
                               const __global int* block_indices,        // [used_blocks]
                               const __global int* block_indices_begins, // [num_seqs+1]
#endif
                               __global float* block_scores,             // [tokens, pitch]
                               const uint num_tiles,
#if QSA_ADDRESS_MODE == 0
                               const uint num_seqs,
#else
                               const uint seq_len,
                               const uint summary_stride,
#endif
                               const uint pitch) {
    const uint HIQ = QSA_IDX_QUERY_HEADS;
    const uint DI = QSA_IDX_HEAD_DIM;
    const uint R = QSA_COMPRESS_RATIO;
    const uint WGN = QSA_SCORE_WG;
    const uint NQ = QSA_SG;
#if QSA_ADDRESS_MODE == 0
    const uint PAGE = QSA_PAGE_SIZE;
    const uint SPP = PAGE / R;
#endif

    __local uint slm_q[QSA_IDX_QUERY_HEADS * QSA_KT][8][QSA_SG];

    const uint wg = get_group_id(0);
    if (wg >= num_tiles) {
        return;
    }
    const uint sg = get_sub_group_id();
    const uint lane = get_sub_group_local_id();

    uint anchor = 0;       // paged: sequence; dense: batch row
    uint token_begin = 0;  // paged: global first token; dense: first position within the row
#if QSA_ADDRESS_MODE == 0
    if (!FUNC_CALL(qsa_resolve_tile_paged)(subsequence_begins, num_seqs, wg, NQ, &anchor, &token_begin)) {
        return;
    }
    const uint seq_begin = (uint)subsequence_begins[anchor];
    const uint seq_end = (uint)subsequence_begins[anchor + 1];
    const uint past_len = (uint)past_lens[anchor];
    const uint page_begin = (uint)block_indices_begins[anchor];
    const uint token0 = token_begin;
    uint valid = seq_end > token0 ? seq_end - token0 : 0;
    const uint first_abs = past_len + token0 - seq_begin;
    const uint n_complete_lane = (lane + first_abs + 1) / R;
#else
    const uint tiles_per_row = (seq_len + NQ - 1) / NQ;
    anchor = wg / tiles_per_row;
    token_begin = (wg % tiles_per_row) * NQ;
    const uint past_len = (uint)past_lens[anchor];
    const uint token0 = anchor * seq_len + token_begin;
    uint valid = seq_len > token_begin ? seq_len - token_begin : 0;
    // Dense: past_lens holds key_length; position = key_length - seq_len + l, left padding sees no block.
    const uint shifted = lane + past_len + token_begin;
    const uint n_complete_lane = shifted < seq_len ? 0 : (shifted + 1 - seq_len) / R;
#endif
    if (valid > NQ) {
        valid = NQ;
    }
    const uint n_blocks_max = valid == 0 ? 0 : sub_group_broadcast(n_complete_lane, valid - 1);

    // Dense-prefix bypass: the finalizer selects every block of such a tile without reading any score.
    if (n_blocks_max <= (uint)QSA_BLOCK_TOPK) {
        return;
    }

    // Stage the query tile as B operands: slm_q[h*KT + k][i][lane] = (Q[lane][16k + 2i], Q[lane][16k + 2i + 1]).
    for (uint t = sg; t < HIQ * QSA_KT; t += WGN) {
        const uint h = t / QSA_KT;
        const uint k = t % QSA_KT;
        uint8 v = (uint8)(0);
        if (lane < valid) {
            v = vload8(0, (const __global uint*)(indexer_q + ((token0 + lane) * HIQ + h) * DI + k * 16));
        }
        slm_q[t][0][lane] = v.s0;
        slm_q[t][1][lane] = v.s1;
        slm_q[t][2][lane] = v.s2;
        slm_q[t][3][lane] = v.s3;
        slm_q[t][4][lane] = v.s4;
        slm_q[t][5][lane] = v.s5;
        slm_q[t][6][lane] = v.s6;
        slm_q[t][7][lane] = v.s7;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint b0 = sg * QSA_MB; b0 < n_blocks_max; b0 += WGN * QSA_MB) {
        ushort rowd[QSA_MB][QSA_KT];
        unroll_for (uint r = 0; r < QSA_MB; r++) {
            const uint b = b0 + r;
            if (b < n_blocks_max) {
#if QSA_ADDRESS_MODE == 0
                // A block past the last visible one has a page id outside this sequence's list: zero it instead.
                const uint page = (uint)block_indices[page_begin + b / SPP];
                const uint src = (page * SPP + b % SPP) * DI;
#else
                const uint src = (anchor * summary_stride + b) * DI;
#endif
                FUNC_CALL(qsa_rd)((const __global ushort*)(summary_cache + src), rowd[r], QSA_KT);
            } else {
                unroll_for (uint k = 0; k < QSA_KT; k++) {
                    rowd[r][k] = 0;
                }
            }
        }

        // The head loop stays rolled: unrolled, the compiler hoists every head's B operands (HIQ * KT int8) into
        // registers ahead of the first mad and spills them.
        float8 acc = (float8)(0.0f);
        __attribute__((opencl_unroll_hint(1)))
        for (uint h = 0; h < HIQ; h++) {
            float8 st = (float8)(0.0f);
            unroll_for (uint k = 0; k < QSA_KT; k++) {
                const short8 a = (short8)((short)rowd[0][k], (short)rowd[1][k], (short)rowd[2][k], (short)rowd[3][k],
                                          (short)rowd[4][k], (short)rowd[5][k], (short)rowd[6][k], (short)rowd[7][k]);
#ifdef cl_intel_subgroup_local_block_io
                const int8 bq = as_int8(intel_sub_group_block_read8(&slm_q[h * QSA_KT + k][0][0]));
#else
                const int8 bq = (int8)(as_int(slm_q[h * QSA_KT + k][0][lane]), as_int(slm_q[h * QSA_KT + k][1][lane]),
                                       as_int(slm_q[h * QSA_KT + k][2][lane]), as_int(slm_q[h * QSA_KT + k][3][lane]),
                                       as_int(slm_q[h * QSA_KT + k][4][lane]), as_int(slm_q[h * QSA_KT + k][5][lane]),
                                       as_int(slm_q[h * QSA_KT + k][6][lane]), as_int(slm_q[h * QSA_KT + k][7][lane]));
#endif
                st = intel_sub_group_f16_f16_matrix_mad_k16(a, bq, st);
            }
            // ReLU per indexer head, strictly before the head reduction.
            acc += fmax(st, (float8)(0.0f));
        }
        acc *= (float)QSA_IDX_SCALE;

        // A row of a dense prefix is never published.
        if (lane < valid && n_complete_lane > (uint)QSA_BLOCK_TOPK) {
            vstore8(acc, 0, block_scores + (size_t)(token0 + lane) * pitch + b0);
        }
    }
}
