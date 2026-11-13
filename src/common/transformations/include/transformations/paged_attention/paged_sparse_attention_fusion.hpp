// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Replaces stateful SparseAttention with PagedSparseAttention.
 *
 * Uses the dimension names from PagedAttentionExtension:
 *   B_token = total new tokens in the call, B_seq = number of sequences,
 *   H = query heads, Hk = key/value heads, S = query/key head size,
 *   Bs = physical cache block size (in tokens).
 * Additional dimensions:
 *   Sv = value head size, Hs = selection heads, KB = selection capacity
 *   (maximum selected sparse blocks per token and selection head).
 * SDPAToPagedAttention must prepare all token inputs before this matcher runs.
 * They use B_token as the leading dimension and a singleton sequence axis
 * inside the graph, as in StateManagementPattern. In particular, the external
 * ngram_embeddings input is [B_token, num_ngrams, embedding_size], restored to
 * [B_token, 1, num_ngrams, embedding_size] for its existing consumers.
 * Bs is distinct from the sparse attention attribute key_block_size;
 * KB counts selected sparse blocks, each containing key_block_size tokens.
 *
 * Matches causal SparseAttention with query_block_size = 1 and stateful K/V
 * caches concatenated with current K/V on the sequence axis:
 *
 *   ReadValue -> (Gather on batch axis) -> Concat(current K/V, axis 2)
 *                                            |
 *   query, block_indices, block_count ------> SparseAttention
 *
 * The replacement packs the current inputs as follows:
 *   query:         [B_token, H, 1, S]   -> [B_token, H * S]
 *   current key:   [B_token, Hk, 1, S]  -> [B_token, Hk * S]
 *   current value: [B_token, Hk, 1, Sv] -> [B_token, Hk * Sv]
 *   block_indices: [B_token, Hs, 1, KB] -> [B_token, Hs, KB]
 *   block_count:   [B_token, Hs, 1]     -> [B_token, Hs]
 * Q/K/V use Transpose followed by Reshape({0, -1}, special_zero = true).
 * Selections use Squeeze on the singleton sequence axis.
 *
 * PagedSparseAttention uses external paged KV caches and scheduling inputs:
 *   key_cache:            [num_blocks, Hk, Bs, S]
 *   value_cache:          [num_blocks, Hk, Bs, Sv]
 *   past_lens:            [B_seq]
 *   subsequence_begins:   [B_seq + 1]
 *   block_indices:        [total_blocks] (physical cache block table)
 *   block_indices_begins: [B_seq + 1]
 * Its selected_block_indices and selected_block_count inputs are the packed
 * SparseAttention selections above; they contain logical sparse block indices
 * and counts, respectively. They are separate from the physical block table.
 * The output is restored from [B_token, H * Sv] to [B_token, H, 1, Sv].
 * The block selection and output gate remain in the graph.
 * Optional scale and sinks are preserved, with an explicit default scale and
 * empty sinks supplied when absent.
 *
 * The enclosing model pass must add pa_params to the model and remove Assign
 * sinks for var_ids_to_remove. Preparing the sparse indexer's paged state is
 * a separate transformation.
 * The replacement uses the internal core PagedSparseAttention operation.
 */

class TRANSFORMATIONS_API PagedSparseAttentionFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedSparseAttentionFusion");
    PagedSparseAttentionFusion(ov::pass::paged_attention::PaParams& pa_params,
                               const ov::pass::paged_attention::Options& options,
                               std::unordered_set<std::string>& var_ids_to_remove);

private:
    size_t m_layer_index = 0;
};

}  // namespace pass
}  // namespace ov
