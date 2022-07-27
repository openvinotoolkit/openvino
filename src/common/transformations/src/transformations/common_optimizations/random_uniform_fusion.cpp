// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/random_uniform_fusion.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::RandomUniformFusion::RandomUniformFusion() {
    MATCHER_SCOPE(RandomUniformFusion);
    const auto data_pattern = ngraph::pattern::any_input();
    const auto ru_min_input_pattern = ngraph::pattern::any_input();
    const auto ru_max_input_pattern = ngraph::pattern::any_input();
    const auto random_uniform_pattern =
        ngraph::pattern::wrap_type<opset8::RandomUniform>({data_pattern, ru_min_input_pattern, ru_max_input_pattern},
                                                          pattern::consumers_count(1));
    const auto const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();

    const auto convert_pattern = ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
    const auto random_uniform_or_convert_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{random_uniform_pattern, convert_pattern});

    const auto mul_add_pattern =
        ngraph::pattern::wrap_type<opset8::Multiply, opset8::Add>({random_uniform_or_convert_pattern, const_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(data_pattern);
        const auto random_uniform = pattern_map.at(random_uniform_pattern);
        const auto constant = pattern_map.at(const_pattern);
        const auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        const auto old_const = std::dynamic_pointer_cast<opset8::Constant>(constant.get_node_shared_ptr());
        if (!old_const)
            return false;
        if (!old_const->get_element_type().is_real())
            return false;

        auto const_shape = old_const->get_shape();
        if (shape_size(const_shape) != 1)
            return false;

        const auto& value = old_const->cast_vector<double>();
        auto new_const = op::Constant::create(ru->get_out_type(), Shape{}, value);

        const auto& mul_add = pattern_map.at(mul_add_pattern);
        const auto mul_add_ptr = std::dynamic_pointer_cast<ngraph::Node>(mul_add.get_node_shared_ptr());
        const auto new_mul_add1 = mul_add_ptr->clone_with_new_inputs({ru->input_value(1), new_const});
        const auto new_mul_add2 = mul_add_ptr->clone_with_new_inputs({ru->input_value(2), new_const});

        const auto& folded_const1 = ngraph::get_constant_from_source(new_mul_add1);
        const auto& folded_const2 = ngraph::get_constant_from_source(new_mul_add2);

        const auto new_ru = ru->clone_with_new_inputs(
            {data, folded_const1 ? folded_const1 : new_mul_add1, folded_const2 ? folded_const2 : new_mul_add2});

        if (pattern_map.count(convert_pattern)) {
            const auto& convert = pattern_map.at(convert_pattern);
            const auto cvt = std::dynamic_pointer_cast<opset8::Convert>(convert.get_node_shared_ptr());
            if (!cvt)
                return false;
            if (!cvt->get_element_type().is_real())
                return false;
            const auto new_ru_conv = cvt->clone_with_new_inputs({new_ru});
            copy_runtime_info({ru, cvt, mul_add.get_node_shared_ptr()},
                              {new_mul_add1, new_mul_add2, new_ru, new_ru_conv});
            new_ru_conv->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), new_ru_conv);
        } else {
            copy_runtime_info({ru, mul_add.get_node_shared_ptr()}, {new_mul_add1, new_mul_add2, new_ru});
            new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), new_ru);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_add_pattern, matcher_name);
    this->register_matcher(m, callback);
}
