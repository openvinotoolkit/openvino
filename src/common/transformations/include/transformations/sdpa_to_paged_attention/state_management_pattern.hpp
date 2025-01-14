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
                           ParameterVector& model_remaining_params,
                           const std::shared_ptr<ov::op::v0::Constant>& sliding_window,
                           ParameterVector& parameters_to_remove,
                           int& layer_index,
                           ov::Output<Node> max_context_len,
                           ParameterVector& block_indices_inputs_for_each_layer,
                           ResultVector& score_results,
                           bool use_per_layer_block_indices_inputs,
                           bool use_score_outputs,
                           bool allow_cache_rotation,
                           ParameterVector& rotated_block_indices_inputs_for_each_layer,
                           ParameterVector& rotation_deltas_inputs_for_each_layer,
                           std::shared_ptr<op::v0::Parameter> model_rotation_trig_lut);
};
