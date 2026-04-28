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
 * @brief Fuses GroupConvolution-based causal Conv1D state update into internal::PagedCausalConv1D.
 *
 * For example, the following graph
 *
 *            +------------------+
 *            | ReadValue (cache)|
 *            +------------------+
 *                      |
 *              [optional Gather]
 *                      |
 *                      +--------------------+
 *                                           |
 * token branch --------------------------> +------------------+
 *                                          | Concat (axis=-1) |
 *                                          +------------------+
 *                                                    |
 *                                                    v
 *                                          +------------------+
 *                                          | GroupConvolution |
 *                                          +------------------+
 *                                                    |
 *                                                    v
 *                                          +------------------+
 *                                          |  Slice (axis=2)  |
 *                                          +------------------+
 *                                                    |
 *                                                    v
 *                                                   ...
 *
 * [optional] Slice(state_concat) -> Result("cache_params.present.conv.*")
 *
 * is transformed to:
 *
 * token branch --> [optional Transpose] --> Reshape ----------+
 *                                                             |
 * conv_state_table.N -----------------------------------------+
 *                                                             |
 * reshaped_weights -------------------------------------------+
 *                                                             |
 * zero_bias --------------------------------------------------+
 *                                                             |
 * subsequence_begins -----------------------------------------+
 * la.block_indices -------------------------------------------+
 * la.block_indices_begins ------------------------------------+
 * la.past_lens -----------------------------------------------+
 * la.cache_interval ------------------------------------------+
 *                                                             v
 *                                         +--------------------------------+
 *                                         | internal::PagedCausalConv1D    |
 *                                         +--------------------------------+
 *                                                         |
 *                                                         v
 *                                             +------------------------+
 *                                             | Unsqueeze (axis = 2)   |
 *                                             +------------------------+
 *                                                         |
 *                                                         v
 *                                                        ...
 *
 * [optional] legacy present-state Result is rewired to preserve external behavior.
 */
class TRANSFORMATIONS_API PagedCausalConv1DFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PagedCausalConv1DFusion");
    PagedCausalConv1DFusion(ov::pass::paged_attention::PaParams& pa_params,
                            std::unordered_set<std::string>& var_ids_to_remove);

private:
    ov::pass::paged_attention::PaParams& m_params;
    std::unordered_set<std::string>& m_var_ids_to_remove;
    static int m_layer_index;
};

}  // namespace pass
}  // namespace ov
