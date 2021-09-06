// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::DropoutWithRandomUniformReplacer, "DropoutWithRandomUniformReplacer", 0);

ngraph::pass::DropoutWithRandomUniformReplacer::DropoutWithRandomUniformReplacer() {
    MATCHER_SCOPE(DropoutWithRandomUniformReplacer);
    const auto shape_of_pattern = ngraph::pattern::wrap_type<opset8::ShapeOf>();
    const auto ru_min_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    const auto ru_max_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    const auto random_uniform_pattern = ngraph::pattern::wrap_type<opset8::RandomUniform>(
        {shape_of_pattern, ru_min_const_pattern, ru_max_const_pattern},
        pattern::consumers_count(1));
    const auto add_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    const auto add_pattern = ngraph::pattern::wrap_type<opset8::Add>({random_uniform_pattern, add_const_pattern});

    const auto floor_pattern = ngraph::pattern::wrap_type<opset8::Floor>({add_pattern});

    const auto mul_input_pattern = ngraph::pattern::any_input();
    const auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({floor_pattern, mul_input_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto & pattern_map = m.get_pattern_value_map();
        const auto random_uniform = pattern_map[random_uniform_pattern];
        const auto shape_of = pattern_map[shape_of_pattern];
        const auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        const auto broadcast_const = opset8::Constant::create(element::f32, Shape{}, {0.5});
        const auto broadcast = register_new_node<opset8::Broadcast>(broadcast_const, shape_of);

        broadcast->set_friendly_name(ru->get_friendly_name());
        copy_runtime_info(ru, broadcast);
        ngraph::replace_node(ru, broadcast);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}
