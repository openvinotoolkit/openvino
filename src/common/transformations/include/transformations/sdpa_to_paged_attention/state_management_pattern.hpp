// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StateManagementPattern;

}  // namespace pass
}  // namespace ov

class ov::pass::StateManagementPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StateManagementPattern");
    StateManagementPattern(ParameterVector& kv_parameters,
                           ParameterVector& model_wide_params,
                           ParameterVector& parameters_to_remove,
                           int& layer_index,
                           ov::Output<Node> max_context_len,
                           ParameterVector& block_indices_inputs_for_each_layer,
                           ResultVector& score_results,
                           bool use_per_layer_block_indices_inputs,
                           bool use_score_outputs,
                           bool allow_cache_rotation,
                           bool allow_score_aggregation,
                           bool allow_xattention,
                           ParameterVector& rotated_block_indices_inputs_for_each_layer,
                           ParameterVector& rotation_deltas_inputs_for_each_layer,
                           ParameterVector& xattention_threshold_inputs_for_each_layer,
                           const std::map<std::string, std::shared_ptr<op::v0::Parameter>>& optional_model_wide_params);
};
