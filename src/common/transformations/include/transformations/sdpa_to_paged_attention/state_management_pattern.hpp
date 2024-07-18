// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class StateManagementPattern;

}  // namespace pass
}  // namespace ov

class ov::pass::StateManagementPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StateManagementPattern", "0");
    StateManagementPattern(ParameterVector& kv_parameters,
                           const ParameterVector& model_remaining_params,
                           const std::shared_ptr<ov::op::v0::Constant>& sliding_window,
                           ParameterVector& parameters_to_remove,
                           int& layer_index,
                           ov::Output<Node> max_context_len,
                           OutputVector& scores_outputs,
                           bool use_cache_eviction);
};