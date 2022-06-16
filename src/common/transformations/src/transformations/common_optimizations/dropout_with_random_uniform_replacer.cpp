// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/pass/pattern/op/or.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::DropoutWithRandomUniformReplacer::DropoutWithRandomUniformReplacer() {
    MATCHER_SCOPE(DropoutWithRandomUniformReplacer);
    const auto shape_pattern = ngraph::pattern::any_input();
    const auto ru_min_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    const auto ru_max_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    const auto random_uniform_pattern =
        ngraph::pattern::wrap_type<opset8::RandomUniform>({shape_pattern, ru_min_const_pattern, ru_max_const_pattern},
                                                          pattern::consumers_count(1));
    const auto convert_pattern = ngraph::pattern::wrap_type<opset8::Convert>({random_uniform_pattern});
    const auto add_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    const auto convert_or_random_uniform_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{convert_pattern, random_uniform_pattern});

    const auto add_pattern =
        ngraph::pattern::wrap_type<opset8::Add>({convert_or_random_uniform_pattern, add_const_pattern});

    const auto floor_pattern = ngraph::pattern::wrap_type<opset8::Floor>({add_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto random_uniform = pattern_map.at(random_uniform_pattern);
        const auto shape_of = pattern_map.at(shape_pattern);
        const auto ru = std::dynamic_pointer_cast<opset8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        auto min_const_value =
            std::dynamic_pointer_cast<opset8::Constant>(pattern_map.at(ru_min_const_pattern).get_node_shared_ptr());
        auto max_const_value =
            std::dynamic_pointer_cast<opset8::Constant>(pattern_map.at(ru_max_const_pattern).get_node_shared_ptr());
        auto add_const_value =
            std::dynamic_pointer_cast<opset8::Constant>(pattern_map.at(add_const_pattern).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value<double>(min_const_value, 0.0) &&
                                     op::util::has_constant_value<double>(max_const_value, 1.0);
        if (!valid_constant_values)
            return false;

        if (!add_const_value)
            return false;

        auto add_const_vector = add_const_value->cast_vector<double>();
        if (add_const_vector.size() > 1)
            return false;

        // Add const should have zero fractional part
        if (add_const_vector[0] - std::round(add_const_vector[0]) != 0.0)
            return false;

        const auto broadcast_const = opset8::Constant::create(ru->get_out_type(), Shape{}, {0.5});
        const auto broadcast = std::make_shared<opset8::Broadcast>(broadcast_const, shape_of);

        broadcast->set_friendly_name(ru->get_friendly_name());
        copy_runtime_info(ru, broadcast);
        ngraph::replace_node(ru, broadcast);
        MATCHER_SCOPE_ENABLE(DropoutWithRandomUniformReplacer);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(floor_pattern, matcher_name);
    this->register_matcher(m, callback);
}
