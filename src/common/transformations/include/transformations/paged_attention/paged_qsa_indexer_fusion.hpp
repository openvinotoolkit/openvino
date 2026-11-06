// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Converts a stateful internal QSAIndexer, as created by FuseQSAIndexer, to internal PagedQSAIndexer.
 *
 *   query, raw keys, cos / sin ──────────────────────────► Reshape [B * L, width] ─┐
 *   q / k norm weights ────────────────────────────────────────────────────────────┤
 *   past_lens, subsequence_begins,                                                 ├─► PagedQSAIndexer
 *   block_indices, block_indices_begins ───────────────────────────────────────────┤      │      │
 *   index_key_cache.N, index_block_rope_cache.N, block_index_keys_cache.N ─────────┘   indices count
 *                                                                                         │      │
 *                                                              Reshape [B, L, S, ...], Transpose to [B, S, L, ...]
 *
 * The selections of PagedQSAIndexer are token-major, so they are restored to the dense [B, S, L, ...] layout of
 * QSAIndexer for the existing consumers.
 * Run before PagedSparseAttentionFusion, which removes these layout adapters from sparse-attention inputs.
 * The paging inputs are shared with PagedAttention, block_indices included, and are added if missing. Each
 * converted indexer gets its own three caches, which share the main KV page table, so their page size is the
 * PagedAttention block size. The dense states must be read from variables, directly or through the beam Gather,
 * in the layout created by FuseQSAIndexer. Their variable ids are added to var_ids_to_remove.
 *
 * Query normalization/rotation flags are preserved, so plugin compilation can absorb query preprocessing later.
 * The query and raw keys must have static head counts and f32, f16 or bf16 data.
 */
class TRANSFORMATIONS_API PagedQSAIndexerFusion : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedQSAIndexerFusion");

    PagedQSAIndexerFusion(paged_attention::PaParams& pa_params, std::unordered_set<std::string>& var_ids_to_remove);

private:
    size_t m_layer_index = 0;
};

}  // namespace ov::pass
