// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Computes the layers after shared-cache PagedAttention only for the rows that are selected.
 *
 * Layers that do not write the KV cache are needed only for rows that reach a row selection on
 * the outputs, such as the sampled_tokens_indices Gather GenAI adds before the LM head. The pass
 * finds the region between the selection and the earlier layers in which each row is computed
 * independently and which contains shared-cache PagedAttention layers. It gathers the selected
 * rows on every tensor entering the region. For each sequence, it keeps the rows from the first
 * selected row to the end, or the last row when none is selected, and derives the kept rows and
 * per-sequence lengths in the graph from the selection, past_lens and subsequence_begins.
 * Tensors with rows are recognized by the symbol of their leading dimension. The model is left
 * unchanged when it is not clear that a tensor entering the region has no rows.
 *
 * Before: rows -> shared PA(Q, K/V, lengths) -> ... -> Gather(indices)
 * After:  Gather(rows, kept) -> shared PA(Q, Gather(K/V, kept), kept lengths) -> ...
 *                            -> Gather(remapped indices)
 */
class TRANSFORMATIONS_API PruneSharedKVRows : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PruneSharedKVRows");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace ov::pass
