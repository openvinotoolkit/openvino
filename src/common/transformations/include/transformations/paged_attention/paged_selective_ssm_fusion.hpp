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

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses SelectiveSSM into internal::PagedSelectiveSSM with paged attention.
 *
 * The stateful SelectiveSSM (Mamba2 selective state-space recurrence) keeps its recurrent
 * state in a Variable that is read through ReadValue (optionally routed through a Gather for
 * beam search). For continuous batching that state has to be managed through a paged block
 * table instead. This pass replaces the matched SelectiveSSM with internal::PagedSelectiveSSM,
 * wiring in the shared paged-attention scheduling parameters and a per-layer state table.
 *
 * For example, the following graph
 *
 *             +------------------+
 *             | recurrent_state  |
 *             | (ReadValue)      |
 *             +------------------+
 *                      |
 *          +---+---+---+--+---+---+
 *          |   |   |   |   |   |
 *          v   v   v   v   v   v
 *    +------------------+
 *    | SelectiveSSM     |
 *    |  (internal op)   |
 *    +------------------+
 *             | |
 *      output0| |output1 (state)
 *             | |
 *             v v
 *           ... [optional Result/Assign for state writeback]
 *
 * is transformed to:
 *
 *    selective_ssm_state_table.N ------------+
 *    A, dt, B, x, C (flattened) -------------+------> internal::PagedSelectiveSSM
 *    subsequence_begins ---------------------+
 *    la.block_indices -----------------------+
 *    la.block_indices_begins ----------------+
 *    la.past_lens (num_processed_tokens) ----+
 *    la.cache_interval ----------------------+
 *                                            v
 *                                  +------------------+
 *                                  |Reshape for output|
 *                                  +------------------+
 *                                            |
 *                                            v
 *                                           ...
 */
class TRANSFORMATIONS_API PagedSelectiveSSMFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedSelectiveSSMFusion");
    PagedSelectiveSSMFusion(ov::pass::paged_attention::PaParams& pa_params,
                            std::unordered_set<std::string>& var_ids_to_remove);

private:
    size_t m_layer_index = 0;
};

}  // namespace pass
}  // namespace ov
