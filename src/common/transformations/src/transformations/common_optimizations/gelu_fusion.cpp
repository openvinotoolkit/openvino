// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/gelu_fusion.hpp"

#include <math.h>

#include <memory>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset9.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::GeluFusionWithErfOne::GeluFusionWithErfOne() {
    MATCHER_SCOPE(GeluFusionWithErfOne);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = ngraph::pattern::any_input();
    auto div_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto div = ngraph::pattern::wrap_type<ngraph::opset7::Divide>({input, div_constant});
    auto erf = ngraph::pattern::wrap_type<ngraph::opset7::Erf>({div});
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({add_constant, erf});
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();

    // (0.5 * x) * (1 + erf(x / sqrt(2))
    auto mul_first = ngraph::pattern::wrap_type<ngraph::opset7::Multiply>({input, mul_constant});
    auto mul = ngraph::pattern::wrap_type<ngraph::opset7::Multiply>({mul_first, add});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, static_cast<float>(M_SQRT2)) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ngraph::opset7::Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(
            {
                pattern_to_output.at(div).get_node_shared_ptr(),
                pattern_to_output.at(erf).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(mul_first).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
            },
            gelu);
        ngraph::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::GeluFusionWithErfTwo::GeluFusionWithErfTwo() {
    MATCHER_SCOPE(GeluFusionWithErfTwo);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = ngraph::pattern::any_input();
    auto div_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto div = ngraph::pattern::wrap_type<ngraph::opset7::Divide>({input, div_constant});
    auto erf = ngraph::pattern::wrap_type<ngraph::opset7::Erf>({div});
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({add_constant, erf});
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();

    // 0.5 * (x * (1 + erf(x / sqrt(2)))
    auto mul_first = ngraph::pattern::wrap_type<ngraph::opset7::Multiply>({input, add});
    auto mul = ngraph::pattern::wrap_type<ngraph::opset7::Multiply>({mul_constant, mul_first});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, static_cast<float>(M_SQRT2)) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ngraph::opset7::Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(
            {
                pattern_to_output.at(div).get_node_shared_ptr(),
                pattern_to_output.at(erf).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(mul_first).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
            },
            gelu);
        ngraph::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::GeluFusionWithErfThree::GeluFusionWithErfThree() {
    MATCHER_SCOPE(GeluFusionWithErfThree);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = ngraph::pattern::any_input();
    auto div_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto div = ngraph::pattern::wrap_type<ngraph::opset7::Divide>({input, div_constant});
    auto erf = ngraph::pattern::wrap_type<ngraph::opset7::Erf>({div});
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({add_constant, erf});
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();

    // x * (0.5 * (1 + erf(x / sqrt(2)))
    auto mul_first = ngraph::pattern::wrap_type<ngraph::opset7::Multiply>({add, mul_constant});
    auto mul = ngraph::pattern::wrap_type<ngraph::opset7::Multiply>({input, mul_first});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
            pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, static_cast<float>(M_SQRT2)) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ngraph::opset7::Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(
            {
                pattern_to_output.at(div).get_node_shared_ptr(),
                pattern_to_output.at(erf).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(mul_first).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
            },
            gelu);
        ngraph::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithErfFour::GeluFusionWithErfFour() {
    MATCHER_SCOPE(GeluFusionWithErfFour);
    using namespace ov;
    using namespace ov::opset9;
    using namespace ov::pass::pattern;

    auto input = any_input();
    auto mul1_constant = wrap_type<Constant>();
    auto mul1 = wrap_type<Multiply>({input, mul1_constant});
    auto erf = wrap_type<Erf>({mul1});
    auto mul2_constant = wrap_type<Constant>();
    auto mul2 = wrap_type<Multiply>({erf, mul2_constant});
    auto add_constant = wrap_type<Constant>();
    auto add = wrap_type<Add>({add_constant, mul2});

    // x * (0.5 + 0.5 * erf(x * (1 / sqrt(2))))
    auto mul3 = wrap_type<Multiply>({input, add});

    matcher_pass_callback callback = [=](Matcher& m) {
        NodeRegistry rg;
        auto pattern_to_output = m.get_pattern_map();
        auto x_output = pattern_to_output.at(input);

        auto mul1_const_value = std::dynamic_pointer_cast<Constant>(pattern_to_output.at(mul1_constant));
        auto add_const_value = std::dynamic_pointer_cast<Constant>(pattern_to_output.at(add_constant));
        auto mul2_const_value = std::dynamic_pointer_cast<Constant>(pattern_to_output.at(mul2_constant));

        if (!mul1_const_value || !add_const_value || !mul2_const_value) {
            return false;
        }

        bool valid_constant_values =
            ngraph::op::util::has_constant_value<float>(mul1_const_value, 1.0f / M_SQRT2, 0.001f) &&
            ngraph::op::util::has_constant_value<float>(add_const_value, 0.5f) &&
            ngraph::op::util::has_constant_value<float>(mul2_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = rg.make<Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), rg.get());
        replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<Matcher>(mul3, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::GeluFusionWithTanh::GeluFusionWithTanh() {
    MATCHER_SCOPE(GeluFusionWithTanh);
    // Replaces a sub-graph with a Gelu (Tanh) op
    // Gaussian Error Linear Unit, TanH based approximation:
    // x * (0.5 * (1 + tanh([sqrt(2 / pi)] * [x + 0.044715^3]))

    auto input = ngraph::pattern::any_input();
    auto pow_constant = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto pow = ngraph::pattern::wrap_type<ngraph::opset9::Power>({input, pow_constant});

    auto mul_0_constant = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto mul_0 = ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({pow, mul_0_constant});

    auto add_0 = ngraph::pattern::wrap_type<ngraph::opset9::Add>({input, mul_0});

    auto mul_1_constant = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto mul_1 = ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({add_0, mul_1_constant});

    auto tanh = ngraph::pattern::wrap_type<ngraph::opset9::Tanh>({mul_1});

    auto add_1_constant = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto add_1 = ngraph::pattern::wrap_type<ngraph::opset9::Add>({tanh, add_1_constant});

    auto mul_2_constant = ngraph::pattern::wrap_type<ngraph::opset9::Constant>();
    auto mul_2 = ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({add_1, mul_2_constant});

    auto mul_3 = ngraph::pattern::wrap_type<ngraph::opset9::Multiply>({input, mul_2});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto pow_constant_value = std::dynamic_pointer_cast<ngraph::opset9::Constant>(
            pattern_to_output.at(pow_constant).get_node_shared_ptr());
        auto mul_0_constant_value = std::dynamic_pointer_cast<ngraph::opset9::Constant>(
            pattern_to_output.at(mul_0_constant).get_node_shared_ptr());
        auto mul_1_constant_value = std::dynamic_pointer_cast<ngraph::opset9::Constant>(
            pattern_to_output.at(mul_1_constant).get_node_shared_ptr());
        auto mul_2_constant_value = std::dynamic_pointer_cast<ngraph::opset9::Constant>(
            pattern_to_output.at(mul_2_constant).get_node_shared_ptr());
        auto add_1_constant_value = std::dynamic_pointer_cast<ngraph::opset9::Constant>(
            pattern_to_output.at(add_1_constant).get_node_shared_ptr());

        if (!pow_constant_value || !add_1_constant_value || !mul_0_constant_value || !mul_1_constant_value ||
            !mul_2_constant_value) {
            return false;
        }

        constexpr float pi = 3.141592653589793238462643383279502884f;
        bool valid_constant_values =
            op::util::has_constant_value<float>(pow_constant_value, 3.0f) &&
            op::util::has_constant_value<float>(mul_0_constant_value, 0.044715f, 0.001f) &&
            op::util::has_constant_value<float>(mul_1_constant_value, std::sqrt(2.0f / pi), 0.01f) &&
            op::util::has_constant_value<float>(mul_2_constant_value, 0.5f) &&
            op::util::has_constant_value<float>(add_1_constant_value, 1.0f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ngraph::opset9::Gelu>(x_output, op::GeluApproximationMode::TANH);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(
            {
                pattern_to_output.at(pow).get_node_shared_ptr(),
                pattern_to_output.at(mul_0).get_node_shared_ptr(),
                pattern_to_output.at(mul_1).get_node_shared_ptr(),
                pattern_to_output.at(mul_2).get_node_shared_ptr(),
                pattern_to_output.at(mul_3).get_node_shared_ptr(),
                pattern_to_output.at(tanh).get_node_shared_ptr(),
                pattern_to_output.at(add_0).get_node_shared_ptr(),
                pattern_to_output.at(add_1).get_node_shared_ptr(),
            },
            gelu);
        ngraph::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_3, matcher_name);
    register_matcher(m, callback);
}
