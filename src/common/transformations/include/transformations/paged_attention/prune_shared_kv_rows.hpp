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
 * @brief Adapts an existing row selection for shared-cache PagedAttention layers.
 *
 * When a shared-cache query follows a common row selection after an earlier attention,
 * the pass replaces the selection indices with a runtime input, gathers the current
 * K/V rows, and uses compacted per-sequence lengths for the affected PA layers.
 * The model must declare shared_kv_row_selection in its runtime info. The runtime
 * must provide the three shared_kv_* inputs: selected rows must be contiguous
 * suffixes within each packed sequence, with corresponding past and subsequence
 * lengths. Models with token_type_ids, query-to-query bias, or ambiguous
 * selections are left unchanged.
 *
 * Before: full rows -> Gather -> shared PA(Q, full K/V, lengths)
 * After:  full rows -> Gather(shared_kv_tokens_indices)
 *                    -> shared PA(Q, Gather(K/V), shared lengths)
 */
class TRANSFORMATIONS_API PruneSharedKVRows : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PruneSharedKVRows");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace ov::pass
