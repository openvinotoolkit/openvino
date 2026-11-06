// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <unordered_set>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Converts causal SparseAttention with paged QSA selections to PagedSparseAttention.
 *
 *   ReadValue ─ [beam Gather] ─ Concat(current K/V) ─┐
 *   query ─────────────────────────────────────────┼─► SparseAttention
 *   PagedQSAIndexer ─ Reshape ─ Transpose ───────────┘
 *                          │
 *                          ▼
 *   query, current K/V ─ Transpose ─ Reshape ───────┐
 *   key_cache.N, value_cache.N ────────────────────┼─► PagedSparseAttention ─ Reshape ─ Transpose
 *   PagedQSAIndexer selections and page table ─────┘
 *
 * Query, key and value use [B, H, L, D] layouts with static head counts and head sizes. The key and value
 * caches are concatenated along the sequence axis and read from variables, directly or through a beam Gather.
 * Selections must be the dense-layout adapters created by PagedQSAIndexerFusion, from the same indexer and
 * with the same key_block_size. Only causal attention with query_block_size = 1 and f32/f16/bf16 data is converted.
 *
 * The operation consumes the indexer's token-major selections and shares its paging metadata. Each converted
 * attention gets key_cache.N and value_cache.N parameters, without colliding with existing PagedAttention caches.
 * Dense KV variable ids are added to var_ids_to_remove for cleanup by SDPAToPagedAttention.
 */
class TRANSFORMATIONS_API PagedSparseAttentionFusion : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedSparseAttentionFusion");

    PagedSparseAttentionFusion(paged_attention::PaParams& pa_params,
                               std::unordered_set<std::string>& var_ids_to_remove);

private:
    size_t m_layer_index = 0;
};

}  // namespace ov::pass
