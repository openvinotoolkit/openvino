// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of the BlockSparseAttention operator.
 *
 * Scaled dot-product attention where the Lq x Lk score matrix is tiled into
 * block_size_q x block_size_kv blocks and only key/value blocks marked `true` in
 * block_mask contribute to the row-wise softmax; masked blocks are skipped entirely
 * (never scored, never included in the softmax denominator) rather than materialized
 * as -inf, matching the op's stated semantics.
 *
 * @param query        Pointer to query,  shape [B, Lq, Hq, E].
 * @param key          Pointer to key,    shape [B, Lk, Hkv, E].
 * @param value        Pointer to value,  shape [B, Lk, Hkv, Ev].
 * @param block_mask   Pointer to boolean block mask, shape [Bm, Hm, num_q_blocks, num_kv_blocks],
 *                     `Bm` in {1, B}, `Hm` in {1, Hkv, Hq}, `true` (nonzero) means "attend".
 * @param output       Pointer to output, shape [B, Lq, Hq, Ev].
 * @param B            Batch size.
 * @param Lq           Query sequence length.
 * @param Lk           Key/value sequence length.
 * @param Hq           Number of query heads.
 * @param Hkv          Number of key/value heads; Hq must be a multiple of Hkv.
 * @param E            Query/key feature dim.
 * @param Ev           Value feature dim.
 * @param Bm           block_mask batch dim (1 or B).
 * @param Hm           block_mask head dim (1, Hkv, or Hq).
 * @param block_size_q  Query block size.
 * @param block_size_kv Key/value block size.
 * @param scale        Softmax scale applied to raw dot products.
 * @param causal       If true, additionally drop key positions j > i (token-level causal mask).
 */
template <typename T>
void block_sparse_attention(const T* query,
                            const T* key,
                            const T* value,
                            const char* block_mask,
                            T* output,
                            size_t B,
                            size_t Lq,
                            size_t Lk,
                            size_t Hq,
                            size_t Hkv,
                            size_t E,
                            size_t Ev,
                            size_t Bm,
                            size_t Hm,
                            size_t block_size_q,
                            size_t block_size_kv,
                            float scale,
                            bool causal) {
    const size_t group = Hq / Hkv;
    const size_t num_kv_blocks = (Lk + block_size_kv - 1) / block_size_kv;
    const size_t num_q_blocks = (Lq + block_size_q - 1) / block_size_q;

    std::vector<float> acc(Ev);
    for (size_t b = 0; b < B; ++b) {
        const size_t mb = (Bm == 1) ? 0 : b;
        for (size_t h = 0; h < Hq; ++h) {
            const size_t hkv = h / group;
            const size_t mh = (Hm == 1) ? 0 : (Hm == Hkv ? hkv : h);
            for (size_t i = 0; i < Lq; ++i) {
                const size_t qb = i / block_size_q;
                const T* q_vec = query + ((b * Lq + i) * Hq + h) * E;
                const char* mask_row = block_mask + ((mb * Hm + mh) * num_q_blocks + qb) * num_kv_blocks;

                // Pass 1: max score across attended keys only, for numerically-stable softmax.
                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t kb = 0; kb < num_kv_blocks; ++kb) {
                    if (!mask_row[kb])
                        continue;
                    const size_t j_begin = kb * block_size_kv;
                    const size_t j_end = std::min(j_begin + block_size_kv, Lk);
                    for (size_t j = j_begin; j < j_end; ++j) {
                        if (causal && j > i)
                            continue;
                        const T* k_vec = key + ((b * Lk + j) * Hkv + hkv) * E;
                        float dot = 0.0f;
                        for (size_t d = 0; d < E; ++d)
                            dot += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
                        max_score = std::max(max_score, dot * scale);
                    }
                }

                // Pass 2: accumulate softmax-weighted value over the same attended keys.
                std::fill(acc.begin(), acc.end(), 0.0f);
                float sum = 0.0f;
                for (size_t kb = 0; kb < num_kv_blocks; ++kb) {
                    if (!mask_row[kb])
                        continue;
                    const size_t j_begin = kb * block_size_kv;
                    const size_t j_end = std::min(j_begin + block_size_kv, Lk);
                    for (size_t j = j_begin; j < j_end; ++j) {
                        if (causal && j > i)
                            continue;
                        const T* k_vec = key + ((b * Lk + j) * Hkv + hkv) * E;
                        float dot = 0.0f;
                        for (size_t d = 0; d < E; ++d)
                            dot += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
                        const float w = std::exp(dot * scale - max_score);
                        sum += w;
                        const T* v_vec = value + ((b * Lk + j) * Hkv + hkv) * Ev;
                        for (size_t d = 0; d < Ev; ++d)
                            acc[d] += w * static_cast<float>(v_vec[d]);
                    }
                }

                T* out_vec = output + ((b * Lq + i) * Hq + h) * Ev;
                const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (size_t d = 0; d < Ev; ++d)
                    out_vec[d] = static_cast<T>(acc[d] * inv_sum);
            }
        }
    }
}

}  // namespace reference
}  // namespace ov