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
 * The stateful SelectiveSSM keeps its recurrent state in a Variable that is read through
 * ReadValue. For continuous batching that state has to be managed through a paged block
 * table instead. This pass replaces the matched SelectiveSSM with internal::PagedSelectiveSSM,
 * wiring in the shared paged-attention scheduling parameters and a per-layer state table.
 *
 * Graph before:
 *
 *                                      (ReadValue)
 *                                 recurrent_state
 *       A   dt   B   x   C                 |
 *       |    |   |   |   |                 |
 *       v    v   v   v   v                 v
 *      +--------------------------------------+
 *      |             SelectiveSSM             |
 *      +--------------------------------------+
 *              |                    |
 *         out0 | y            state | out1
 *              v                    v
 *                                 Assign
 *
 * is transformed to
 *
 * After - PagedSelectiveSSM, recurrent state kept in a paged block table:
 *
 *     A                                  per-head decay rates [num_heads]
 *     dt, B, x, C                        flattened [batch, len, ...] -> [tokens, ...]
 *     selective_ssm_state_table.N        paged state table, updated in place
 *     subsequence_begins                 per-sequence token spans
 *     la.block_indices                   logical -> physical block rows
 *     la.block_indices_begins            per-sequence block spans
 *     la.past_lens                       num_processed_tokens
 *     la.cache_interval                  state snapshot interval
 *              |  (11 inputs)
 *              v
 *      +--------------------------------------+
 *      |          PagedSelectiveSSM           |   (11 inputs, 1 output)
 *      +--------------------------------------+
 *              |
 *         out0 | y
 *              v
 *           Reshape  -> restore [batch, len, num_heads, head_dim]
 *              v
 *             ...
 */
class TRANSFORMATIONS_API PagedSelectiveSSMFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedSelectiveSSMFusion");
    PagedSelectiveSSMFusion(ov::pass::paged_attention::PaParams& pa_params,
                            std::unordered_set<std::string>& var_ids_to_remove);

    // Number of SelectiveSSM nodes converted to PagedSelectiveSSM once the pass has run.
    size_t get_fused_count() const {
        return m_layer_index;
    }

private:
    size_t m_layer_index = 0;
};

}  // namespace pass
}  // namespace ov
