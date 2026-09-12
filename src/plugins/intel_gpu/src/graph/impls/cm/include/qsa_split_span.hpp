// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Private copy of the scalar Q3 helpers from qsa_sparse_attention.cm, not a shared
// refactor of that kernel. Functions are prefixed qsa_split_ (rather than reusing
// qsa_sparse_attention.cm's exact names) because every QSA stage - including the
// always-present scalar qsa_sparse_attention.cm, which defines the unprefixed names
// at file scope - is batched into ONE clBuildProgram call; reusing the same names
// would fail to compile as a duplicate definition. Bodies are otherwise identical.
#ifndef QSA_SPLIT_SPAN_HPP
#define QSA_SPLIT_SPAN_HPP
#include "qsa_score_common.hpp"
#define QSA_LOG2E 1.4426950408889634f

CM_INLINE void qsa_split_softmax_step(float score, float& running_max, float& running_sum, float& corr, float& weight) {
    const float new_max = running_max > score ? running_max : score;
    vector<float, 2> e;
    e[0] = (running_max - new_max) * QSA_LOG2E;
    e[1] = (score - new_max) * QSA_LOG2E;
    e = cm_exp(e);
    corr = e[0];
    weight = e[1];
    running_sum = running_sum * corr + weight;
    running_max = new_max;
}

template <int DH, int DV, int HKV, int PA_BLOCK, int SPAN>
CM_INLINE void qsa_split_attend_span(vector_ref<float, DH> q,
                                     vector_ref<float, DV> acc,
                                     float& running_max,
                                     float& running_sum,
                                     const half* key_cache ATTR,
                                     const half* value_cache ATTR,
                                     const int32_t* block_indices ATTR,
                                     uint page_begin,
                                     uint kv_head,
                                     uint pos0) {
    const uint page = (uint)block_indices[page_begin + pos0 / (uint)PA_BLOCK];
    const uint tok0 = pos0 % (uint)PA_BLOCK;

    vector<half, SPAN * DH> ktile;
    qsa_gr::load_hv<SPAN * DH>(ktile, key_cache, qsa::kv_cache_offset(page, kv_head, tok0, HKV, PA_BLOCK, DH));
    vector<half, SPAN * DV> vtile;
    qsa_gr::load_hv<SPAN * DV>(vtile, value_cache, qsa::kv_cache_offset(page, kv_head, tok0, HKV, PA_BLOCK, DV));

#pragma unroll
    for (int s = 0; s < SPAN; s++) {
        float score = 0.0f;
#pragma unroll
        for (int k = 0; k < DH; k += 16) {
            vector<float, 16> kv = ktile.template select<16, 1>(s * DH + k);
            score += cm_sum<float>(q.template select<16, 1>(k) * kv);
        }

        float corr = 0.0f;
        float weight = 0.0f;
        qsa_split_softmax_step(score, running_max, running_sum, corr, weight);

#pragma unroll
        for (int k = 0; k < DV; k += 16) {
            vector<float, 16> vv = vtile.template select<16, 1>(s * DV + k);
            acc.template select<16, 1>(k) = acc.template select<16, 1>(k) * corr + vv * weight;
        }
    }
}
#endif  // QSA_SPLIT_SPAN_HPP