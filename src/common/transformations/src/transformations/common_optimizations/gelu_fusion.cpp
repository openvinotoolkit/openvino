// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/gelu_fusion.hpp"

#include <math.h>

#include <cmath>
#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::GeluFusionWithErfOne::GeluFusionWithErfOne() {
    MATCHER_SCOPE(GeluFusionWithErfOne);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = pass::pattern::any_input();
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({input, div_constant});
    auto erf = ov::pass::pattern::wrap_type<ov::op::v0::Erf>({div});
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({add_constant, erf});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();

    // (0.5 * x) * (1 + erf(x / sqrt(2))
    auto mul_first = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_constant});
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({mul_first, add});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, static_cast<float>(M_SQRT2), 0.001f) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::op::v7::Gelu>(x_output);

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithErfTwo::GeluFusionWithErfTwo() {
    MATCHER_SCOPE(GeluFusionWithErfTwo);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = pass::pattern::any_input();
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({input, div_constant});
    auto erf = ov::pass::pattern::wrap_type<ov::op::v0::Erf>({div});
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({add_constant, erf});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();

    // 0.5 * (x * (1 + erf(x / sqrt(2)))
    auto mul_first = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, add});
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({mul_constant, mul_first});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, static_cast<float>(M_SQRT2), 0.001f) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::op::v7::Gelu>(x_output);

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithErfThree::GeluFusionWithErfThree() {
    MATCHER_SCOPE(GeluFusionWithErfThree);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = pass::pattern::any_input();
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({input, div_constant});
    auto erf = ov::pass::pattern::wrap_type<ov::op::v0::Erf>({div});
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({add_constant, erf});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();

    // x * (0.5 * (1 + erf(x / sqrt(2)))
    auto mul_first = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add, mul_constant});
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_first});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto div_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());
        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        if (!div_const_value || !add_const_value || !mul_const_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(div_const_value, static_cast<float>(M_SQRT2), 0.001f) &&
            op::util::has_constant_value<float>(add_const_value, 1.0f) &&
            op::util::has_constant_value<float>(mul_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::op::v7::Gelu>(x_output);

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithErfFour::GeluFusionWithErfFour() {
    MATCHER_SCOPE(GeluFusionWithErfFour);
    using namespace ov;
    using namespace ov::pass::pattern;

    auto input = any_input();
    auto mul1_constant = wrap_type<ov::op::v0::Constant>();
    auto mul1 = wrap_type<ov::op::v1::Multiply>({input, mul1_constant});
    auto erf = wrap_type<ov::op::v0::Erf>({mul1});
    auto mul2_constant = wrap_type<ov::op::v0::Constant>();
    auto mul2 = wrap_type<ov::op::v1::Multiply>({erf, mul2_constant});
    auto add_constant = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({add_constant, mul2});

    // x * (0.5 + 0.5 * erf(x * (1 / sqrt(2))))
    auto mul3 = wrap_type<ov::op::v1::Multiply>({input, add});

    matcher_pass_callback callback = [=](Matcher& m) {
        NodeRegistry rg;
        auto pattern_to_output = m.get_pattern_map();
        auto x_output = pattern_to_output.at(input);

        auto mul1_const_value = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul1_constant));
        auto add_const_value = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant));
        auto mul2_const_value = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul2_constant));

        if (!mul1_const_value || !add_const_value || !mul2_const_value) {
            return false;
        }

        constexpr auto sqrt2 = static_cast<float>(M_SQRT2);
        bool valid_constant_values = ov::op::util::has_constant_value<float>(mul1_const_value, 1.0f / sqrt2, 0.001f) &&
                                     ov::op::util::has_constant_value<float>(add_const_value, 0.5f) &&
                                     ov::op::util::has_constant_value<float>(mul2_const_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = rg.make<ov::op::v7::Gelu>(x_output);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        copy_runtime_info(m.get_matched_nodes(), rg.get());
        replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<Matcher>(mul3, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithTanh::GeluFusionWithTanh() {
    MATCHER_SCOPE(GeluFusionWithTanh);
    // Replaces a sub-graph with a Gelu (ov::op::v0::Tanh) op
    // Gaussian Error Linear Unit, TanH based approximation:
    // x * (0.5 * (1 + tanh([sqrt(2 / pi)] * [x + 0.044715^3]))

    auto input = pass::pattern::any_input();
    auto pow_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto pow = ov::pass::pattern::wrap_type<ov::op::v1::Power>({input, pow_constant});

    auto mul_0_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_0 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({pow, mul_0_constant});

    auto add_0 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, mul_0});

    auto mul_1_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_0, mul_1_constant});

    auto tanh = ov::pass::pattern::wrap_type<ov::op::v0::Tanh>({mul_1});

    auto add_1_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add_1 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({tanh, add_1_constant});

    auto mul_2_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();

    // x * (0.5 * (1 + tanh))
    auto mul_2_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_1, mul_2_constant});
    auto mul_3_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_2_1});

    // (x * 0.5) * (1 + tanh)
    auto mul_2_2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_2_constant});
    auto mul_3_2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_1, mul_2_2});

    auto mul_3 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_3_1, mul_3_2});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto pow_constant_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(pow_constant).get_node_shared_ptr());
        auto mul_0_constant_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_0_constant).get_node_shared_ptr());
        auto mul_1_constant_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_1_constant).get_node_shared_ptr());
        auto mul_2_constant_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_2_constant).get_node_shared_ptr());
        auto add_1_constant_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_1_constant).get_node_shared_ptr());
        if (!pow_constant_value || !add_1_constant_value || !mul_0_constant_value || !mul_1_constant_value ||
            !mul_2_constant_value) {
            return false;
        }

        bool valid_constant_values =
            op::util::has_constant_value<float>(pow_constant_value, 3.0f) &&
            op::util::has_constant_value<float>(mul_0_constant_value, 0.044715f, 0.001f) &&
            op::util::has_constant_value<double>(mul_1_constant_value, std::sqrt(2.0 / M_PI), 0.01) &&
            op::util::has_constant_value<float>(mul_2_constant_value, 0.5f) &&
            op::util::has_constant_value<float>(add_1_constant_value, 1.0f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::op::v7::Gelu>(x_output, op::GeluApproximationMode::TANH);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());

        std::vector<std::shared_ptr<ov::Node>> pattern_nodes =
            {pow, mul_0, mul_1, tanh, add_0, add_1, mul_2_1, mul_2_2, mul_3_1, mul_3_2};
        std::vector<std::shared_ptr<ov::Node>> cp_rt_info_nodes;
        for (const auto& pattern_node : pattern_nodes) {
            if (pattern_to_output.count(pattern_node)) {
                cp_rt_info_nodes.push_back(pattern_to_output.at(pattern_node).get_node_shared_ptr());
            }
        }
        ov::copy_runtime_info(cp_rt_info_nodes, gelu);

        ov::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_3, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithTanhNoPower::GeluFusionWithTanhNoPower() {
    // Replaces a sub-graph with a Gelu (ov::op::v0::Tanh) op
    // x * 0.5 * (1 + tanh((x * 0.044715 * x + 1) * x * sqrt(2 / pi)))
    MATCHER_SCOPE(GeluFusionWithTanhNoPower);
    auto input = pattern::any_input();

    auto const1 = pattern::wrap_type<ov::op::v0::Constant>();
    auto mul1 = pattern::wrap_type<ov::op::v1::Multiply>({input, const1});

    auto mul2 = pattern::wrap_type<ov::op::v1::Multiply>({mul1, input});

    auto const2 = pattern::wrap_type<ov::op::v0::Constant>();
    auto add1 = pattern::wrap_type<ov::op::v1::Add>({const2, mul2});

    auto const3 = pattern::wrap_type<ov::op::v0::Constant>();
    auto mul3 = pattern::wrap_type<ov::op::v1::Multiply>({input, const3});

    auto mul4 = pattern::wrap_type<ov::op::v1::Multiply>({add1, mul3});

    auto tanh = pattern::wrap_type<ov::op::v0::Tanh>({mul4});

    auto const4 = pattern::wrap_type<ov::op::v0::Constant>();
    auto add2 = pattern::wrap_type<ov::op::v1::Add>({tanh, const4});

    auto const5 = pattern::wrap_type<ov::op::v0::Constant>();
    auto mul5 = pattern::wrap_type<ov::op::v1::Multiply>({input, const5});

    auto mul6 = pattern::wrap_type<ov::op::v1::Multiply>({add2, mul5});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto const1_value = pattern_to_output.at(const1).get_node_shared_ptr();
        auto const2_value = pattern_to_output.at(const2).get_node_shared_ptr();
        auto const3_value = pattern_to_output.at(const3).get_node_shared_ptr();
        auto const4_value = pattern_to_output.at(const4).get_node_shared_ptr();
        auto const5_value = pattern_to_output.at(const5).get_node_shared_ptr();

        bool valid_constant_values = op::util::has_constant_value<float>(const1_value, 0.044715f, 0.001f) &&
                                     op::util::has_constant_value<float>(const2_value, 1.0f) &&
                                     op::util::has_constant_value<double>(const3_value, std::sqrt(2.0 / M_PI), 0.01) &&
                                     op::util::has_constant_value<float>(const4_value, 1.0f) &&
                                     op::util::has_constant_value<float>(const5_value, 0.5f);

        if (!valid_constant_values) {
            return false;
        }

        auto gelu = std::make_shared<ov::op::v7::Gelu>(x_output, op::GeluApproximationMode::TANH);

        gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(mul1).get_node_shared_ptr(),
                pattern_to_output.at(mul2).get_node_shared_ptr(),
                pattern_to_output.at(add1).get_node_shared_ptr(),
                pattern_to_output.at(mul3).get_node_shared_ptr(),
                pattern_to_output.at(mul4).get_node_shared_ptr(),
                pattern_to_output.at(tanh).get_node_shared_ptr(),
                pattern_to_output.at(add2).get_node_shared_ptr(),
                pattern_to_output.at(mul5).get_node_shared_ptr(),
                pattern_to_output.at(mul6).get_node_shared_ptr(),
            },
            gelu);
        ov::replace_node(m.get_match_root(), gelu);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mul6, matcher_name);
    this->register_matcher(m, callback);
}
