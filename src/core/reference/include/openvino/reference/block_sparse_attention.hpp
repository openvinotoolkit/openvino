// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov::reference {

/// \brief Reference implementation of BlockSparseAttention-17.
///
/// Unlike a graph built from existing ops (Gather + Reshape + ScaledDotProductAttention),
/// this loop never materializes a gathered/transposed copy of key or value: for every
/// selected block it reads directly from the original `key`/`value` buffers at the
/// computed offset. That is the property the op exists for -- the cost of this loop
/// scales with the amount of *selected* (sparse) work, not with a gather-then-transpose
/// step whose cost scales with `batch * heads * num_query_blocks * gathered_length`
/// regardless of how little was actually selected.
///
/// \param query                [B, H,  L, E]
/// \param key                  [B, Hk, S, E]              (Hk == H or Hk == 1, GQA/MQA style)
/// \param value                [B, Hk, S, Ev]
/// \param block_indices        [B, Hb, L / block_size, k_blocks] (Hb == H or Hb == 1)
/// \param block_indices_mask   same shape as block_indices, non-null only when provided;
///                             a zero byte marks a padding entry that must be ignored.
/// \param scale                pointer to a single scale value, or nullptr for the
///                             default `1 / sqrt(E)`.
/// \param output               [B, H, L, Ev]
template <typename T, typename TIndex>
void block_sparse_attention(const T* query,
                            const T* key,
                            const T* value,
                            const TIndex* block_indices,
                            const char* block_indices_mask,
                            const T* scale,
                            T* output,
                            bool causal,
                            int64_t block_size,
                            const Shape& query_shape,
                            const Shape& key_shape,
                            const Shape& value_shape,
                            const Shape& block_indices_shape) {
    const int64_t B = static_cast<int64_t>(query_shape[0]);
    const int64_t H = static_cast<int64_t>(query_shape[1]);
    const int64_t L = static_cast<int64_t>(query_shape[2]);
    const int64_t E = static_cast<int64_t>(query_shape[3]);
    const int64_t Hk = static_cast<int64_t>(key_shape[1]);
    const int64_t S = static_cast<int64_t>(key_shape[2]);
    const int64_t Ev = static_cast<int64_t>(value_shape[3]);
    const int64_t Hb = static_cast<int64_t>(block_indices_shape[1]);
    const int64_t num_q_blocks = static_cast<int64_t>(block_indices_shape[2]);
    const int64_t k_blocks = static_cast<int64_t>(block_indices_shape[3]);
    const int64_t num_kv_blocks = S / block_size;

    const T scale_val = scale ? *scale : static_cast<T>(1.0 / std::sqrt(static_cast<double>(E)));
    // key/value and block_indices/mask may each provide a single shared head that broadcasts to
    // every query head (Hk == 1 / Hb == 1), or a full per-query-head tensor (Hk == H / Hb == H).
    // Shape inference (block_sparse_attention_shape_inference.hpp) only accepts these two cases,
    // matching the level of head broadcasting ScaledDotProductAttention itself supports; real
    // GQA models expand key/value to the full head count with Broadcast/Tile before attention.
    const auto broadcast_head = [](int64_t h, int64_t dim_size) {
        return dim_size == 1 ? int64_t{0} : h;
    };

    const auto query_at = [&](int64_t b, int64_t h, int64_t l) {
        return query + ((b * H + h) * L + l) * E;
    };
    const auto key_at = [&](int64_t b, int64_t h, int64_t s) {
        return key + ((b * Hk + broadcast_head(h, Hk)) * S + s) * E;
    };
    const auto value_at = [&](int64_t b, int64_t h, int64_t s) {
        return value + ((b * Hk + broadcast_head(h, Hk)) * S + s) * Ev;
    };
    const auto output_at = [&](int64_t b, int64_t h, int64_t l) {
        return output + ((b * H + h) * L + l) * Ev;
    };
    const auto block_indices_at = [&](int64_t b, int64_t h, int64_t qb) {
        return block_indices + ((b * Hb + broadcast_head(h, Hb)) * num_q_blocks + qb) * k_blocks;
    };
    const auto block_indices_mask_at = [&](int64_t b, int64_t h, int64_t qb) -> const char* {
        return block_indices_mask
                   ? block_indices_mask + ((b * Hb + broadcast_head(h, Hb)) * num_q_blocks + qb) * k_blocks
                   : nullptr;
    };

    // Per-row scratch: at most k_blocks*block_size candidate keys contribute to one query row.
    std::vector<T> scores(static_cast<size_t>(k_blocks) * static_cast<size_t>(block_size));
    std::vector<int64_t> positions(scores.size());

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t qb = 0; qb < num_q_blocks; ++qb) {
                const TIndex* bi_row = block_indices_at(b, h, qb);
                const char* mask_row = block_indices_mask_at(b, h, qb);
                for (int64_t qi = 0; qi < block_size; ++qi) {
                    const int64_t q_pos = qb * block_size + qi;
                    const T* q_ptr = query_at(b, h, q_pos);

                    size_t valid_count = 0;
                    T max_score = std::numeric_limits<T>::lowest();
                    for (int64_t kb = 0; kb < k_blocks; ++kb) {
                        if (mask_row && mask_row[kb] == 0) {
                            continue;  // padding entry (ragged top-k selection) -- excluded, not just biased
                        }
                        const int64_t blk = static_cast<int64_t>(bi_row[kb]);
                        if (blk < 0 || blk >= num_kv_blocks) {
                            continue;  // defensive: ignore out-of-range indices instead of reading OOB memory
                        }
                        // token-level causal mask *within* the selected block: a block on or
                        // straddling the diagonal is only partially valid.
                        const int64_t tokens_in_block =
                            causal ? std::min<int64_t>(block_size, q_pos - blk * block_size + 1) : block_size;
                        for (int64_t ki = 0; ki < tokens_in_block; ++ki) {
                            const int64_t k_pos = blk * block_size + ki;
                            const T* k_ptr = key_at(b, h, k_pos);
                            T dot = T{0};
                            for (int64_t e = 0; e < E; ++e) {
                                dot += q_ptr[e] * k_ptr[e];
                            }
                            scores[valid_count] = dot * scale_val;
                            positions[valid_count] = k_pos;
                            max_score = std::max(max_score, scores[valid_count]);
                            ++valid_count;
                        }
                    }

                    T* out_ptr = output_at(b, h, q_pos);
                    std::fill(out_ptr, out_ptr + Ev, T{0});
                    if (valid_count == 0) {
                        continue;  // no valid candidate for this row (e.g. fully-masked edge case)
                    }
                    T sum_exp = T{0};
                    for (size_t i = 0; i < valid_count; ++i) {
                        scores[i] = static_cast<T>(std::exp(static_cast<double>(scores[i] - max_score)));
                        sum_exp += scores[i];
                    }
                    for (size_t i = 0; i < valid_count; ++i) {
                        const T weight = scores[i] / sum_exp;
                        const T* v_ptr = value_at(b, h, positions[i]);
                        for (int64_t e = 0; e < Ev; ++e) {
                            out_ptr[e] += weight * v_ptr[e];
                        }
                    }
                }
            }
        }
    }
}

}  // namespace ov::reference
