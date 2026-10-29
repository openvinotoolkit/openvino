// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PagedQSAFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Converts a stateful Qwen Sparse Attention layer into its paged form: PagedQSAIndexer + PagedSparseAttention.
 *
 * The stateful layer
 *
 *     index query (RMSNorm + RoPE, see QSAIndexQueryBlock) ---------------------------+
 *     QSAPrepare(raw_index_keys, ReadValue(block keys), key_length, cos, sin,          |
 *                k_norm_weight, ReadValue(incomplete keys), ReadValue(incomplete rope)) |
 *                                                    |                                  |
 *                                         SparseAttentionIndexer <---------------------+
 *                                                    |
 *     SparseAttention(q, Concat(ReadValue(K), k), Concat(ReadValue(V), v), block_indices, block_count, key_length
 *                     [, scale [, sinks]])
 *
 * with the QSAPrepare state outputs and the key / value concatenations written back by Assign, becomes
 *
 *     PagedQSAIndexer(raw index query, raw_index_keys, cos, sin, q_norm_weight, k_norm_weight,
 *                     past_lens, subsequence_begins, block_indices, block_indices_begins,
 *                     index_key_cache.N, index_block_rope_cache.N, block_index_keys_cache.N)
 *                                                    |
 *     PagedSparseAttention(q, k, v, key_cache.N, value_cache.N, past_lens, subsequence_begins, block_indices,
 *                          block_indices_begins, selected_block_indices, selected_block_count, scale, sinks)
 *
 * on the token stream: the [N, ..., L, ...] activations are flattened to [N * L, ...], as SDPAToPagedAttention makes
 * every token a sequence of its own (N = tokens, L = 1). The five caches share the page table of the main KV cache;
 * their element type and block size are set by ConvertPagedAttnInputs. The variables of the five states are added to
 * `var_ids_to_remove`.
 */
class ov::pass::PagedQSAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedQSAFusion");
    PagedQSAFusion(ov::pass::paged_attention::PaParams& pa_params, std::unordered_set<std::string>& var_ids_to_remove);
};
