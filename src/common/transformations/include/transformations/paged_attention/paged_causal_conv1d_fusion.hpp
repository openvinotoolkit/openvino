// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "openvino/pass/backward_graph_rewrite.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses GroupConvolution-based causal Conv1D state update into internal::PagedCausalConv1D.
 *
 * Supports two history layouts for a window of (kernel_size - 1) * dilation + 1:
 * a full-window history followed by output trimming, and a window-minus-one history
 * without output trimming. The paged state table always stores the full window.
 * Traverses backwards to match optional output trimming and bias before the convolution alone.
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
 *                                          | [Slice(axis=2)] |
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
 * The replaced state's variable ID is recorded so that the enclosing paged-attention
 * pipeline can remove its Assign. Explicit present-state Results are left intact.
 */
class TRANSFORMATIONS_API PagedCausalConv1DFusion : public ov::pass::BackwardGraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("PagedCausalConv1DFusion");
    PagedCausalConv1DFusion(ov::pass::paged_attention::PaParams& pa_params,
                            std::unordered_set<std::string>& var_ids_to_remove);

private:
    size_t m_layer_index = 0;
};

}  // namespace pass
}  // namespace ov
