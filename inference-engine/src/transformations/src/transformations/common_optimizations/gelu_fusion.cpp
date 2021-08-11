// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/gelu_fusion.hpp"

#include <math.h>

#include <memory>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ov::pass::GeluFusion, "GeluFusion", 0);

NGRAPH_RTTI_DEFINITION(ov::pass::GeluFusionWithErfOne,
                       "GeluFusionWithErfOne", 0);

ov::pass::GeluFusionWithErfOne::GeluFusionWithErfOne() {
    MATCHER_SCOPE(GeluFusionWithErfOne);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = ov::pattern::any_input();
    auto div_constant = ov::pattern::wrap_type<ov::opset7::Constant>();
    auto div = ov::pattern::wrap_type<ov::opset7::Divide>(
        {input, div_constant});
    auto erf = ov::pattern::wrap_type<ov::opset7::Erf>({div});
    auto add_constant = ov::pattern::wrap_type<ov::opset7::Constant>();
    auto add =
        ov::pattern::wrap_type<ov::opset7::Add>({add_constant, erf});
    auto mul_constant = ov::pattern::wrap_type<ov::opset7::Constant>();

    // (0.5 * x) * (1 + erf(x / sqrt(2))
    auto mul_first = ov::pattern::wrap_type<ov::opset7::Multiply>(
        {input, mul_constant});
    auto mul =
        ov::pattern::wrap_type<ov::opset7::Multiply>({mul_first, add});

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, M_SQRT2) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::opset7::Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(div).get_node_shared_ptr(),
                pattern_to_output.at(erf).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(mul_first).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
            },
            gelu);
        ov::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::GeluFusionWithErfTwo,
                       "GeluFusionWithErfTwo", 0);

ov::pass::GeluFusionWithErfTwo::GeluFusionWithErfTwo() {
    MATCHER_SCOPE(GeluFusionWithErfTwo);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = ov::pattern::any_input();
    auto div_constant = ov::pattern::wrap_type<ov::opset7::Constant>();
    auto div = ov::pattern::wrap_type<ov::opset7::Divide>(
        {input, div_constant});
    auto erf = ov::pattern::wrap_type<ov::opset7::Erf>({div});
    auto add_constant = ov::pattern::wrap_type<ov::opset7::Constant>();
    auto add =
        ov::pattern::wrap_type<ov::opset7::Add>({add_constant, erf});
    auto mul_constant = ov::pattern::wrap_type<ov::opset7::Constant>();

    // 0.5 * (x * (1 + erf(x / sqrt(2)))
    auto mul_first =
        ov::pattern::wrap_type<ov::opset7::Multiply>({input, add});
    auto mul = ov::pattern::wrap_type<ov::opset7::Multiply>(
        {mul_constant, mul_first});

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, M_SQRT2) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::opset7::Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(div).get_node_shared_ptr(),
                pattern_to_output.at(erf).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(mul_first).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
            },
            gelu);
        ov::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::GeluFusionWithErfThree,
                       "GeluFusionWithErfThree", 0);

ov::pass::GeluFusionWithErfThree::GeluFusionWithErfThree() {
    MATCHER_SCOPE(GeluFusionWithErfThree);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = ov::pattern::any_input();
    auto div_constant = ov::pattern::wrap_type<ov::opset7::Constant>();
    auto div = ov::pattern::wrap_type<ov::opset7::Divide>(
        {input, div_constant});
    auto erf = ov::pattern::wrap_type<ov::opset7::Erf>({div});
    auto add_constant = ov::pattern::wrap_type<ov::opset7::Constant>();
    auto add =
        ov::pattern::wrap_type<ov::opset7::Add>({add_constant, erf});
    auto mul_constant = ov::pattern::wrap_type<ov::opset7::Constant>();

    // x * (0.5 * (1 + erf(x / sqrt(2)))
    auto mul_first = ov::pattern::wrap_type<ov::opset7::Multiply>(
        {add, mul_constant});
    auto mul = ov::pattern::wrap_type<ov::opset7::Multiply>(
        {input, mul_first});

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value =
            std::dynamic_pointer_cast<ov::opset7::Constant>(
                pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, M_SQRT2) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::opset7::Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(div).get_node_shared_ptr(),
                pattern_to_output.at(erf).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(mul_first).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
            },
            gelu);
        ov::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}
