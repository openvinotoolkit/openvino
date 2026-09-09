/*******************************************************************************
 * Copyright (c) 2018-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

// Indexer block scoring shared by the three Q2 kernels.
//
// Numerical rules that must not be reordered (design section 6.3):
//   1. per-head dot product, then per-head ReLU, then head reduction;
//   2. the 1/sqrt(Di) scale is applied AFTER the ReLU sum, not per head;
//   3. dot products accumulate in FP32 (pooling was already FP32 in Q1).

// Classic include guard - see qsa_gr_common.hpp for why `#pragma once` is not usable here.
#ifndef QSA_SCORE_COMMON_HPP
#define QSA_SCORE_COMMON_HPP

#include "qsa_common.hpp"

#define QSA_SUMMARIES_PER_PAGE (QSA_PA_BLOCK_SIZE / QSA_COMPRESS_RATIO)

/// Number of QSA blocks that are FULLY visible to a query at absolute position `i`.
/// The block-causal condition is p_b + r - 1 <= i, i.e. (b + 1) * r <= i + 1.
/// Partially visible blocks never take part in top-k; their visible tokens survive
/// only through the tail path in Q3.
CM_INLINE uint qsa_num_complete_blocks(uint abs_pos) {
    return (abs_pos + 1) / (uint)QSA_COMPRESS_RATIO;
}

/// Loads the whole indexer query of one token into registers.
///
/// Decode Q2 is the QSA bandwidth main term (~8 MB per layer per step, ~63% of the
/// decode traffic), and the query is reused by EVERY block of the scan, so it must be
/// hoisted out of the block loop. Re-reading it per block would multiply the query
/// traffic by n_block.
template <int HIDX, int DI>
CM_INLINE void qsa_load_query(matrix_ref<half, HIDX, DI> q, const half* indexer_q ATTR, uint token_idx) {
#pragma unroll
    for (int h = 0; h < HIDX; h++) {
        qsa_gr::load_hv<DI>(q.row(h), indexer_q, (token_idx * (uint)HIDX + h) * (uint)DI);
    }
}

CM_INLINE float qsa_block_score(matrix_ref<half, QSA_IDX_HEADS, QSA_IDX_HEAD_DIM> q,
                                const half* summary_cache ATTR,
                                const int32_t* block_indices ATTR,
                                uint page_begin,
                                uint qsa_block) {
    constexpr uint HIDX = QSA_IDX_HEADS;
    constexpr uint DI = QSA_IDX_HEAD_DIM;
    constexpr uint SPP = QSA_SUMMARIES_PER_PAGE;

    const uint page = (uint)block_indices[page_begin + qsa_block / SPP];
    const uint src = qsa::summary_offset(page, qsa_block % SPP, SPP, DI);

    // One summary is DI * 2 bytes of contiguous memory: fetch it with a single wide
    // message, not DI/16 narrow ones.
    vector<half, DI> summary;
    qsa_gr::load_hv<DI>(summary, summary_cache, src);

    float total = 0.0f;
#pragma unroll
    for (uint h = 0; h < HIDX; h++) {
        float dot = 0.0f;
#pragma unroll
        for (uint k = 0; k < DI; k += 16) {
            vector<float, 16> qv = q.row(h).template select<16, 1>(k);
            vector<float, 16> kv = summary.template select<16, 1>(k);
            dot += cm_sum<float>(qv * kv);
        }
        total += dot > 0.0f ? dot : 0.0f;  // ReLU per indexer head, before reduction
    }
    return total * (float)QSA_IDX_SCALE;  // 1 / sqrt(Di), applied after the ReLU sum
}

/// Resolves the sequence a flattened token belongs to and its absolute position.
CM_INLINE void qsa_resolve_token(const int32_t* subsequence_begins ATTR,
                                 const int32_t* past_lens ATTR,
                                 uint num_seqs,
                                 uint token_idx,
                                 uint& seq,
                                 uint& abs_pos) {
    seq = 0;
    for (uint i = 0; i < num_seqs; i++) {
        if (token_idx >= (uint)subsequence_begins[i] && token_idx < (uint)subsequence_begins[i + 1]) {
            seq = i;
            break;
        }
    }
    abs_pos = (uint)past_lens[seq] + token_idx - (uint)subsequence_begins[seq];
}

#endif  // QSA_SCORE_COMMON_HPP
