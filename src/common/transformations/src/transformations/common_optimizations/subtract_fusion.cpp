// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/subtract_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::SubtractFusion::SubtractFusion() {
    MATCHER_SCOPE(SubtractFusion);
    auto p_input = pattern::any_input();

    auto p_mul_const = pattern::wrap_type<opset8::Constant>();
    auto p_mul = pattern::wrap_type<opset8::Multiply>({p_input, p_mul_const});

    auto p_neg = pattern::wrap_type<opset8::Negative>({p_input});

    auto p_mul_or_neg = std::make_shared<pattern::op::Or>(OutputVector({p_mul, p_neg}));

    auto p_add_input = pattern::any_input();
    auto p_add = ngraph::pattern::wrap_type<opset8::Add>({p_add_input, p_mul_or_neg});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        const auto& minuend_input = pattern_to_output.at(p_add_input);
        const auto& subtrahend_input = pattern_to_output.at(p_input);

        const auto& add = pattern_to_output.at(p_add).get_node_shared_ptr();

        NodeVector nodes_to_replace{add};

        if (pattern_to_output.count(p_mul_const)) {
            auto minus_one_const =
                std::dynamic_pointer_cast<opset8::Constant>(pattern_to_output.at(p_mul_const).get_node_shared_ptr());
            if (!op::util::has_constant_value<float>(minus_one_const, -1.)) {
                return false;
            }
            nodes_to_replace.emplace_back(pattern_to_output.at(p_mul).get_node_shared_ptr());
        } else {
            nodes_to_replace.emplace_back(pattern_to_output.at(p_neg).get_node_shared_ptr());
        }

        auto sub = register_new_node<opset8::Subtract>(minuend_input, subtrahend_input);
        sub->set_friendly_name(add->get_friendly_name());
        copy_runtime_info(nodes_to_replace, sub);
        replace_node(add, sub);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_add, matcher_name);
    register_matcher(m, callback);
}
