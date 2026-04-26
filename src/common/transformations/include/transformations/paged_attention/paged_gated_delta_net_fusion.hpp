// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "openvino/pass/pass.hpp"
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
class TRANSFORMATIONS_API PagedGatedDeltaNetFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PagedGatedDeltaNetFusion");
    PagedGatedDeltaNetFusion(ov::pass::paged_attention::PaParams& pa_params,
                             const ov::pass::paged_attention::Options& options,
                             std::unordered_set<std::string>& var_ids_to_remove);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ov::pass::paged_attention::PaParams& m_params;
    const ov::pass::paged_attention::Options& m_options;
    std::unordered_set<std::string>& m_var_ids_to_remove;
};

}  // namespace pass
}  // namespace ov
