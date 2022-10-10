// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/divide_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::DivideFusion::DivideFusion() {
    MATCHER_SCOPE(DivideFusion);
    auto p_pow_input = pattern::any_input();
    auto p_pow_const = pattern::wrap_type<opset8::Constant>();
    auto p_pow = pattern::wrap_type<opset8::Power>({p_pow_input, p_pow_const});
    auto p_mul_input = pattern::any_input();
    auto p_mul = ngraph::pattern::wrap_type<opset8::Multiply>({p_mul_input, p_pow});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        const auto& minuend_input = pattern_to_output.at(p_mul_input);
        const auto& subtrahend_input = pattern_to_output.at(p_pow_input);
        const auto& mul = pattern_to_output.at(p_mul).get_node_shared_ptr();
        const auto& pow = pattern_to_output.at(p_pow).get_node_shared_ptr();
        const auto& minus_one = pattern_to_output.at(p_pow_const).get_node_shared_ptr();

        auto minus_one_const = std::dynamic_pointer_cast<opset8::Constant>(minus_one);
        if (!minus_one_const || !op::util::has_constant_value<float>(minus_one_const, -1.)) {
            return false;
        }

        auto div = register_new_node<opset8::Divide>(minuend_input, subtrahend_input);
        div->set_friendly_name(mul->get_friendly_name());
        copy_runtime_info({mul, pow}, div);
        replace_node(mul, div);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_mul, matcher_name);
    register_matcher(m, callback);
}
