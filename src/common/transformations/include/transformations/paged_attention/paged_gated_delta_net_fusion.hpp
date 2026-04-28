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
 * @brief Fuses GatedDeltaNet into internal::PagedGatedDeltaNet with paged attention.
 *
 * For example, the following graph
 *
 *             +------------------+
 *             | recurrent_state  |
 *             | (Parameter)      |
 *             +------------------+
 *                      |
 *            +---+---+-+--+---+
 *            |   |   |   |   |
 *            v   v   v   v   v
 *    +------------------+
 *    | GatedDeltaNet    |
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
 *    gated_delta_state_table.N -----------+
 *    gate, beta (flattened) -----+        |
 *    query, key, value (flat) ---+------> | internal::PagedGatedDeltaNet
 *    subsequence_begins ----------+        |
 *    la.block_indices -----------+        |
 *    la.* (runtime params) ------+        |
 *                                         v
 *                              +------------------+
 *                              |Reshape for output|
 *                              +------------------+
 *                                         |
 *                                         v
 *                                        ...
 */
class TRANSFORMATIONS_API PagedGatedDeltaNetFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedGatedDeltaNetFusion");
    PagedGatedDeltaNetFusion(ov::pass::paged_attention::PaParams& pa_params,
                             std::unordered_set<std::string>& var_ids_to_remove);

private:
    size_t m_layer_index = 0;
};

}  // namespace pass
}  // namespace ov
