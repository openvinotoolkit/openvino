// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "propagate_optimal_bs.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

using namespace ov::intel_cpu;
NGRAPH_RTTI_DEFINITION(PropagateOptimalBS, "PropagateOptimalBS", 0);

ov::intel_cpu::PropagateOptimalBS::PropagateOptimalBS() {
    auto root = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto node = m.get_match_root();
        if (has_optimal_bs(node) || is_type<ngraph::opset1::Result>(node))
            return false;
        const auto& pshape = m.get_match_value().get_partial_shape();
        if (pshape.size() == 0)
            return false;

        auto set_parent_opt_bs = [&node](const std::shared_ptr<ov::Node>& parent) {
            if (!has_optimal_bs(parent)) {
                return false;
            }
            const auto parent_bs = get_optimal_bs(parent);
            set_optimal_bs(node, parent_bs);
            return true;
        };

        const auto node_batch = pshape[0].get_length();
        for (const auto& input : node->input_values()) {
            const auto& input_ps = input.get_partial_shape();
            if (!input_ps.rank().is_static() || input_ps.size() == 0 || input_ps[0].get_length() != node_batch)
                continue;

            if (set_parent_opt_bs(input.get_node_shared_ptr())) {
                break;
            }
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, "PropagateOptimalBS");
    this->register_matcher(m, callback);
}
