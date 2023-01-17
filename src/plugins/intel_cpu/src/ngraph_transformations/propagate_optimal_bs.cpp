// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_optimal_bs.hpp"

#include <openvino/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

using namespace ov::intel_cpu;
NGRAPH_RTTI_DEFINITION(PropagateOptimalBS, "PropagateOptimalBS", 0);

ov::intel_cpu::PropagateOptimalBS::PropagateOptimalBS() {
    auto root = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto node = m.get_match_root();
        if (has_optimal_bs(node) || is_type<ov::opset1::Result>(node))
            return false;

        // TODO: do we really need it?
        if (is_type<ov::opset1::Reshape>(node) || is_type<ov::opset1::Concat>(node) ||
            is_type<ov::opset1::Split>(node) || is_type<ov::opset1::VariadicSplit>(node))
            return false;

        const auto& pshape = m.get_match_value().get_partial_shape();
        if (pshape.size() == 0)
            return false;

        auto set_parent_opt_bs = [&node](const std::shared_ptr<ov::Node>& parent) {
            if (!has_optimal_bs(parent))
                return false;
            const auto parent_bs = get_optimal_bs(parent);
            set_optimal_bs(node, parent_bs);
            return true;
        };

        const auto node_batch = pshape[0].get_length();
        for (const auto& input : node->input_values()) {
            const auto& input_ps = input.get_partial_shape();
            if (input_ps.is_dynamic() || input_ps.size() == 0 || input_ps[0].get_length() != node_batch)
                continue;
            if (set_parent_opt_bs(input.get_node_shared_ptr()))
                break;
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "PropagateOptimalBS");
    this->register_matcher(m, callback);
}
