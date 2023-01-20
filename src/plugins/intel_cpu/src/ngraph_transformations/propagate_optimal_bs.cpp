// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_optimal_bs.hpp"

#include <openvino/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

using namespace ov::intel_cpu;

bool PropagateOptimalBS::run_on_model(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (has_optimal_bs(node) || is_type<ov::opset1::Result>(node) || node->get_input_size() == 0)
            continue;

        // don't propagate an optimal batch size through the layers that could set hardcoded output shapes
        if (is_type<ov::opset1::Interpolate>(node) || is_type<ov::opset4::Interpolate>(node) ||
            is_type<ov::opset1::Reshape>(node) || is_type<ov::opset1::StridedSlice>(node) ||
            is_type<ov::opset8::Gather>(node)) {
            continue;
        }

        auto set_parent_opt_bs = [&node](const std::shared_ptr<ov::Node>& parent) {
            if (!has_optimal_bs(parent))
                return false;
            const auto parent_bs = get_optimal_bs(parent);
            set_optimal_bs(node, parent_bs);
            return true;
        };

        const size_t batch_dim = 0;
        const auto& pshape = node->get_output_partial_shape(0);
        if (pshape.size() <= batch_dim)
            continue;

        const auto node_batch = pshape[batch_dim].get_length();
        for (const auto& input : node->input_values()) {
            const auto& input_ps = input.get_partial_shape();
            if (input_ps.is_dynamic() || input_ps.size() <= batch_dim || input_ps[batch_dim].get_length() != node_batch)
                continue;
            if (set_parent_opt_bs(input.get_node_shared_ptr()))
                break;
        }
    }
    return false;
}
