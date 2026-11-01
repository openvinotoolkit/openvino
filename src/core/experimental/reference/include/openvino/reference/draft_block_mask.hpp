// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of the DraftBlockMask operator.
 *
 * Mean-pools q_windows/k_windows per block, scores block pairs per head, folds in a
 * precomputed static local-window bias, softmaxes, and keeps only the top-k scoring key
 * blocks per (head, temporal-block) row.
 *
 * NOTE: assumes the query/key block ordering is temporal-major, i.e. block index `qi = t * HW + s`
 * for temporal block `t` in [0, seqlen) and spatial block `s` in [0, HW) (respectively
 * `kj = t' * HW + s'` for key, `t'` in [0, Nk / HW)). `local_mask` is indexed by the spatial
 * component only (`qi % HW`, `kj % HW`) and tiled across the temporal dimension, matching the
 * cached shape/attribute-only mask built once per (h, w, local_range) in the source model. `HW` is
 * a single value -- local_mask's own (square) extent, passed in explicitly by the caller -- since
 * the K-side temporal group count (`Nk / HW`) is independent of `seqlen` in general (e.g. a
 * streaming decoder's accumulated key/value cache growing past the current query window); it must
 * NOT be re-derived as `Nk / seqlen`.
 *
 * @param q_windows   Pointer to query blocks, shape [Nq, block_s, D].
 * @param k_windows   Pointer to key blocks, shape [Nk, block_s, D].
 * @param local_mask  Pointer to boolean local-window bias, shape [HW, HW] (square; HW is
 *                     local_mask's own extent, not derived from Nq/Nk/seqlen).
 * @param out         Pointer to boolean output mask, shape [batch_size, num_heads, Nq, Nk].
 * @param Nq          Number of query blocks.
 * @param Nk          Number of key blocks.
 * @param block_s     Window size (tokens per block).
 * @param D           Total feature dim (num_heads * head_dim).
 * @param num_heads   Number of attention heads.
 * @param HW          local_mask's own (square) extent. Nq == seqlen * HW; Nk need only be a
 *                     multiple of HW.
 * @param seqlen      Number of temporal blocks (Nq == seqlen * HW).
 * @param topk        Number of key blocks kept per query row.
 * @param batch_size  Output batch dimension; the mask itself is batch-independent and tiled.
 */
template <typename T>
void draft_block_mask(const T* q_windows,
                      const T* k_windows,
                      const char* local_mask,
                      char* out,
                      size_t Nq,
                      size_t Nk,
                      size_t block_s,
                      size_t D,
                      size_t num_heads,
                      size_t HW,
                      size_t seqlen,
                      size_t topk,
                      size_t batch_size) {
    const size_t head_dim = D / num_heads;
    // HW is local_mask's own square extent, passed explicitly by the caller -- never derived from
    // Nq/seqlen or Nk/seqlen. Nq == seqlen * HW is guaranteed by shape_infer()/evaluate(); Nk need
    // only be a multiple of HW (Nk / HW, the K-side temporal group count, is independent of seqlen
    // -- e.g. a streaming decoder's accumulated KV cache growing past the current query window).

    // 1. Mean-pool each block over its window axis: [N, block_s, D] -> [N, D].
    std::vector<float> q_mean(Nq * D, 0.0f);
    std::vector<float> k_mean(Nk * D, 0.0f);
    auto mean_pool = [block_s, D](const T* windows, size_t n, std::vector<float>& mean) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t s = 0; s < block_s; ++s) {
                const T* row = windows + (i * block_s + s) * D;
                for (size_t d = 0; d < D; ++d) {
                    mean[i * D + d] += static_cast<float>(row[d]);
                }
            }
            for (size_t d = 0; d < D; ++d) {
                mean[i * D + d] /= static_cast<float>(block_s);
            }
        }
    };
    mean_pool(q_windows, Nq, q_mean);
    mean_pool(k_windows, Nk, k_mean);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> attn(num_heads * Nq * Nk);

    // 2. Per-head block x block similarity + local mask bias + softmax.
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t qi = 0; qi < Nq; ++qi) {
            const size_t q_local = qi % HW;
            const float* q_vec = q_mean.data() + qi * D + h * head_dim;
            float* row = attn.data() + (h * Nq + qi) * Nk;

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t kj = 0; kj < Nk; ++kj) {
                const size_t k_local = kj % HW;
                const float* k_vec = k_mean.data() + kj * D + h * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                float score = dot * scale;
                // local_mask is [HW, HW] row-major: row stride is HW.
                if (!local_mask[q_local * HW + k_local]) {
                    score = -std::numeric_limits<float>::infinity();
                }
                row[kj] = score;
                max_score = std::max(max_score, score);
            }

            float sum = 0.0f;
            for (size_t kj = 0; kj < Nk; ++kj) {
                const float e = std::isfinite(row[kj]) ? std::exp(row[kj] - max_score) : 0.0f;
                row[kj] = e;
                sum += e;
            }
            for (size_t kj = 0; kj < Nk; ++kj) {
                row[kj] = sum > 0.0f ? row[kj] / sum : 0.0f;
            }
        }
    }

    // 3. Top-k threshold per (head, temporal-block) row, flattened over (spatial_q, Nk).
    std::vector<char> mask(num_heads * Nq * Nk, 0);
    const size_t row_elems = HW * Nk;
    const size_t k_clamped = std::min(topk, row_elems > 0 ? row_elems - 1 : size_t(0));
    std::vector<float> flat(row_elems);
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t t = 0; t < seqlen; ++t) {
            const float* base = attn.data() + (h * Nq + t * HW) * Nk;
            std::copy(base, base + row_elems, flat.begin());
            std::nth_element(flat.begin(), flat.begin() + k_clamped, flat.end(), std::greater<float>());
            const float threshold = flat[k_clamped];
            for (size_t e = 0; e < row_elems; ++e) {
                mask[(h * Nq + t * HW) * Nk + e] = base[e] > threshold ? 1 : 0;
            }
        }
    }

    // 4. Tile the (head-major) mask across the batch dimension.
    const size_t per_batch = num_heads * Nq * Nk;
    for (size_t b = 0; b < batch_size; ++b) {
        std::copy(mask.begin(), mask.end(), out + b * per_batch);
    }
}

}  // namespace reference
}  // namespace ov