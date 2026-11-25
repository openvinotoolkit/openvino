// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

inline constexpr char shared_kv_row_selection_key[] = "shared_kv_row_selection";

/**
 * @ingroup ov_transformation_common_api
 * @brief Computes shared-cache PagedAttention layers only for the rows that are sampled.
 *
 * Shared-cache layers do not write the KV cache, so only rows that reach sampled logits are
 * needed. The model must declare shared_kv_row_selection in its runtime info, select rows after
 * an earlier attention, and have the sampled_tokens_indices input. For each sequence, the pass
 * keeps the rows from the first sampled row to the end, or the last row when none is sampled.
 * It derives the kept rows and the per-sequence lengths from sampled_tokens_indices, past_lens
 * and subsequence_begins, and remaps sampled_tokens_indices to the kept rows. Models are left
 * unchanged when they have token_type_ids, query-to-query bias, or ambiguous selections, when
 * shared-layer rows reach an output other than through sampled_tokens_indices, or when
 * shared-layer scores are used.
 *
 * Before: full rows -> Gather -> shared PA(Q, full K/V, lengths) -> Gather(sampled_tokens_indices)
 * After:  full rows -> Gather(kept rows) -> shared PA(Q, Gather(K/V), kept lengths)
 *                   -> Gather(remapped sampled_tokens_indices)
 */
class TRANSFORMATIONS_API PruneSharedKVRows : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PruneSharedKVRows");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace ov::pass
