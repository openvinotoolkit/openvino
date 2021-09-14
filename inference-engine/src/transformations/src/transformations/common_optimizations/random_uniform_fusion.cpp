// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/random_uniform_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformFusion, "RandomUniformFusion", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformMulFusion, "RandomUniformMulFusion", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::RandomUniformAddFusion, "RandomUniformAddFusion", 0);

ngraph::pass::RandomUniformMulFusion::RandomUniformMulFusion() {
    MATCHER_SCOPE(RandomUniformMulFusion);
    const auto data_pattern = ngraph::pattern::any_input();
    const auto ru_min_input_pattern = ngraph::pattern::any_input();
    const auto ru_max_input_pattern = ngraph::pattern::any_input();
    const auto random_uniform_pattern =
        ngraph::pattern::wrap_type<opset8::RandomUniform>({data_pattern, ru_min_input_pattern, ru_max_input_pattern},
                                                          pattern::consumers_count(1));
    const auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();

    const auto convert_pattern = ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
    const auto random_uniform_or_convert_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{random_uniform_pattern, convert_pattern});

    const auto mul_pattern =
        ngraph::pattern::wrap_type<opset8::Multiply>({random_uniform_or_convert_pattern, mul_const_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(data_pattern);
        const auto random_uniform = pattern_map.at(random_uniform_pattern);
        const auto mul_constant = pattern_map.at(mul_const_pattern);
        const auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        const auto mul_const = std::dynamic_pointer_cast<opset8::Constant>(mul_constant.get_node_shared_ptr());
        if (!mul_const)
            return false;
        if (!mul_const->get_element_type().is_real())
            return false;

        auto const_shape = mul_const->get_shape();
        if (shape_size(const_shape) != 1)
            return false;
        const auto & value = mul_const->cast_vector<double>();
        auto new_mul_const = op::Constant::create(ru->get_out_type(), Shape{}, value);

        if (pattern_map.count(convert_pattern)) {
            const auto& mul = pattern_map.at(mul_pattern);
            const auto ml = std::dynamic_pointer_cast<opset8::Multiply>(mul.get_node_shared_ptr());
            if (!ml)
                return false;

            const auto& convert = pattern_map.at(convert_pattern);
            const auto cvt = std::dynamic_pointer_cast<opset8::Convert>(convert.get_node_shared_ptr());
            if (!cvt)
                return false;
            if (!cvt->get_element_type().is_real())
                return false;
            const auto new_mul1 = ml->clone_with_new_inputs({ru->input_value(1), new_mul_const});
            const auto new_mul2 = ml->clone_with_new_inputs({ru->input_value(2), new_mul_const});
            const auto new_ru = ru->clone_with_new_inputs({data, new_mul1, new_mul2});
            new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
            const auto new_ru_conv = cvt->clone_with_new_inputs({new_ru});
            copy_runtime_info({ru, cvt}, {new_ru, new_ru_conv});
            ngraph::replace_node(m.get_match_root(), new_ru_conv);

        } else {
            const auto& mul = pattern_map.at(mul_pattern);
            const auto ml = std::dynamic_pointer_cast<opset8::Multiply>(mul.get_node_shared_ptr());
            if (!ml)
                return false;
            const auto new_mul1 = ml->clone_with_new_inputs({ru->input_value(1), new_mul_const});
            const auto new_mul2 = ml->clone_with_new_inputs({ru->input_value(2), new_mul_const});
            const auto new_ru = ru->clone_with_new_inputs({data, new_mul1, new_mul2});
            new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
            copy_runtime_info({mul.get_node_shared_ptr(), ru}, {new_mul1, new_mul2, new_ru});
            ngraph::replace_node(m.get_match_root(), new_ru);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::RandomUniformAddFusion::RandomUniformAddFusion() {
    MATCHER_SCOPE(RandomUniformAddFusion);
    const auto data_pattern = ngraph::pattern::any_input();
    const auto ru_min_input_pattern = ngraph::pattern::any_input();
    const auto ru_max_input_pattern = ngraph::pattern::any_input();
    const auto random_uniform_pattern =
        ngraph::pattern::wrap_type<opset8::RandomUniform>({data_pattern, ru_min_input_pattern, ru_max_input_pattern},
                                                          pattern::consumers_count(1));
    const auto add_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();

    const auto convert_pattern = ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
    const auto convert_or_random_uniform_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{convert_pattern, random_uniform_pattern});
    const auto add_pattern =
        ngraph::pattern::wrap_type<opset8::Add>({convert_or_random_uniform_pattern, add_const_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto data = pattern_map.at(data_pattern);
        const auto random_uniform = pattern_map.at(random_uniform_pattern);
        const auto add_constant = pattern_map.at(add_const_pattern);
        const auto ru_max_input = pattern_map.at(ru_max_input_pattern);
        const auto ru_min_input = pattern_map.at(ru_min_input_pattern);
        const auto add = pattern_map.at(add_pattern);
        const auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(random_uniform.get_node_shared_ptr());

        const auto add_const = std::dynamic_pointer_cast<opset8::Constant>(add_constant.get_node_shared_ptr());
        if (!add_const)
            return false;
        if (!add_const->get_element_type().is_real())
            return false;

        auto const_shape = add_const->get_shape();
        size_t const_shape_size = shape_size(const_shape);

        if (const_shape_size != 1)
            return false;
        const auto value = add_const->cast_vector<double>();
        auto new_add_const = op::Constant::create(ru->get_out_type(), Shape{}, value);

        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        if (pattern_map.count(convert_pattern)) {
            const auto& convert = pattern_map.at(convert_pattern);
            const auto cvt = std::dynamic_pointer_cast<opset8::Convert>(convert.get_node_shared_ptr());
            if (!cvt)
                return false;
            if (!cvt->get_element_type().is_real())
                return false;
            const auto add1 = std::make_shared<ngraph::opset8::Add>(ru_min_input, new_add_const);
            const auto add2 = std::make_shared<ngraph::opset8::Add>(ru_max_input, new_add_const);
            const auto new_ru = ru->clone_with_new_inputs({data, add1, add2});
            new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
            const auto new_ru_conv = cvt->clone_with_new_inputs({new_ru});
            copy_runtime_info({add.get_node_shared_ptr(), ru, cvt}, {new_ru, new_ru_conv});
            ngraph::replace_node(m.get_match_root(), new_ru_conv);
        } else {
            const auto add1 = std::make_shared<ngraph::opset8::Add>(ru_min_input, new_add_const);
            const auto add2 = std::make_shared<ngraph::opset8::Add>(ru_max_input, new_add_const);
            const auto new_ru = ru->clone_with_new_inputs({data, add1, add2});
            new_ru->set_friendly_name(m.get_match_root()->get_friendly_name());
            copy_runtime_info({add.get_node_shared_ptr(), ru}, new_ru);
            ngraph::replace_node(m.get_match_root(), new_ru);
        }
        return true;
    };

    const auto m = std::make_shared<ngraph::pattern::Matcher>(add_pattern, matcher_name);
    this->register_matcher(m, callback);
}
