// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q1 - QSAIndexer (non-paged, functional state) prepare.
//
// Same maths as qsa_indexer_prepare_paged.cl with dense addressing: batch IS the sequence axis, the flat token index
// is t = b * seq_len + l, and the staged incomplete block / its first-token rope row live in distinct input and output
// state buffers (the host has already copied every state input into its output).
//
// JIT: QSA_IDX_QUERY_HEADS, QSA_IDX_HEAD_DIM, QSA_IDX_ROTARY_DIM, QSA_COMPRESS_RATIO, QSA_IDX_NORM_EPS, SUBGROUP_SIZE
// Dispatch: gws = [SG, HIQ + 1, num_wg], lws = [SG, 1, 1]

#define QSA_INDEXER_HELPERS
#include "qsa_common.cl"

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(qsa_indexer_prepare_dense)(const __global half* q_idx,           // [B, L, Hiq, Di]
                                  const __global half* k_idx,           // [B, L, 1, Di]
                                  const __global half* q_norm_weight,   // [Di]
                                  const __global half* k_norm_weight,   // [Di]
                                  const __global half* rope_cos,        // [B, L, rotary_dim]
                                  const __global half* rope_sin,        // [B, L, rotary_dim]
                                  const __global int* key_length,       // [B]
                                  const __global half* raw_in,          // [B, 1, r-1, Di]
                                  const __global half* rope_in,         // [B, 2*rotary_dim]
                                  __global half* summary_cache,         // [B, 1, floor(S_cap/r), Di]
                                  __global half* raw_out,               // [B, 1, r-1, Di]
                                  __global half* rope_out,              // [B, 2*rotary_dim]
                                  __global half* indexer_q,             // [tokens, Hiq, Di]
                                  const uint num_wg,
                                  const uint blocks_per_row,
                                  const uint seq_len,
                                  const uint summary_stride) {
    const uint HIQ = QSA_IDX_QUERY_HEADS;
    const uint DI = QSA_IDX_HEAD_DIM;
    const uint ROT = QSA_IDX_ROTARY_DIM;
    const uint R = QSA_COMPRESS_RATIO;

    const uint wg = get_group_id(2);
    const uint role = get_group_id(1);
    const uint lane = get_sub_group_local_id();
    if (wg >= num_wg) {
        return;
    }

    // key_length lives in GPU memory: work-group `slot` of row `batch` takes block past/r + slot.
    const uint batch = wg / blocks_per_row;
    const uint total_len = (uint)key_length[batch];
    const uint n_new = total_len < seq_len ? total_len : seq_len;
    const uint past_len = total_len - n_new;
    const uint qsa_block = past_len / R + wg % blocks_per_row;
    if (qsa_block * R >= total_len) {
        return;
    }

    // Flat token index of logical position `past_len` (the first real row of this call).
    const uint seq_begin = batch * seq_len + (seq_len - n_new);

    const uint block_start = qsa_block * R;
    const uint block_end = block_start + R;
    const bool block_completes = block_end <= total_len;
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

    if (!block_completes) {
        // Stage the keys of the new incomplete block (slot p % r) and its first-token rope row.
        for (uint p = lo; p < hi; p++) {
            const uint token_idx = seq_begin + (p - past_len);
            ushort kh[QSA_NE];
            FUNC_CALL(qsa_rd)((const __global ushort*)(k_idx + token_idx * DI), kh, QSA_NE);
            FUNC_CALL(qsa_wr)((__global ushort*)(raw_out + (batch * (R - 1) + (p - block_start)) * DI), kh, QSA_NE);
#if QSA_IDX_ROTARY_DIM > 0
            if (p == block_start) {
                for (uint i = lane; i < ROT; i += QSA_SG) {
                    rope_out[batch * 2 * ROT + 2 * i] = rope_cos[token_idx * ROT + i];
                    rope_out[batch * 2 * ROT + 2 * i + 1] = rope_sin[token_idx * ROT + i];
                }
            }
#endif
        }
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
            // Staged by an earlier call; slot r-1 is never staged (it completes the block).
            FUNC_CALL(qsa_rd_f32)(raw_in + (batch * (R - 1) + s) * DI, v, QSA_NE);
        }
        unroll_for (uint j = 0; j < QSA_NE; j++) {
            acc[j] += v[j];
        }
    }
    unroll_for (uint j = 0; j < QSA_NE; j++) {
        acc[j] = convert_float(convert_half_rte(acc[j] * (1.0f / (float)R)));
    }
    FUNC_CALL(qsa_rms_norm)(acc, k_norm_weight);
#if QSA_IDX_ROTARY_DIM > 0
    if (block_start >= past_len) {
        const uint first_token = seq_begin + (block_start - past_len);
        FUNC_CALL(qsa_rope)(acc, rope_cos + first_token * ROT, rope_sin + first_token * ROT, 1);
    } else {
        FUNC_CALL(qsa_rope)(acc, rope_in + batch * 2 * ROT, rope_in + batch * 2 * ROT + 1, 2);
    }
#endif
    FUNC_CALL(qsa_wr_f32)(summary_cache + (batch * summary_stride + qsa_block) * DI, acc, QSA_NE);
}
