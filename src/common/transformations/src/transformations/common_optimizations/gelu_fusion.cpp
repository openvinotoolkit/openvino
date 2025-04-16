// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/gelu_fusion.hpp"

#include <math.h>

#include <cmath>
#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
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

using namespace ov;
using namespace ov::op::util;
using namespace ov::pass::pattern::op;

constexpr auto SQRT2 = static_cast<float>(M_SQRT2);
constexpr auto SQRT1_2 = static_cast<float>(M_SQRT1_2);
constexpr auto SQRT_2_PI = 0.79788456080286535588f;  // std::sqrt(M_2_PI)

namespace {

Predicate check_value(float ref, float eps = std::numeric_limits<float>::epsilon()) {
    return Predicate(
        [=](const Output<Node>& output) -> bool {
            return ov::op::util::has_constant_value<float>(output.get_node_shared_ptr(), ref, eps);
        },
        "has_constant_value(" + std::to_string(ref) + ")");
}

bool gelu_replacer(ov::pass::pattern::Matcher& m,
                   const std::shared_ptr<ov::Node>& pattern_input_to_gelu,
                   ov::op::GeluApproximationMode mode = ov::op::GeluApproximationMode::ERF) {
    ov::pass::NodeRegistry rg;
    auto pattern_to_output = m.get_pattern_value_map();
    auto x_output = pattern_to_output.at(pattern_input_to_gelu);

    auto gelu = rg.make<ov::op::v7::Gelu>(x_output, mode);

    gelu->set_friendly_name(m.get_match_root()->get_friendly_name());
    copy_runtime_info(m.get_matched_nodes(), rg.get());
    replace_node(m.get_match_root(), gelu);
    return true;
}

}  // namespace

ov::pass::GeluFusionWithErfOne::GeluFusionWithErfOne() {
    MATCHER_SCOPE(GeluFusionWithErfOne);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = pass::pattern::any_input();
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT2, 0.001f));
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({input, div_constant});

    // In case of ConvertDivideWithConstant is applied and Div is converted to Mul
    auto mul_as_div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT1_2, 0.001f));
    auto mul_as_div = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_as_div_constant});
    auto erf_input = std::make_shared<Or>(ov::OutputVector{div, mul_as_div});

    auto erf = ov::pass::pattern::wrap_type<ov::op::v0::Erf>({erf_input});

    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(1.0f));
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({add_constant, erf});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(0.5f));

    // (0.5 * x) * (1 + erf(x / sqrt(2))
    auto mul_first = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_constant});
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({mul_first, add});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        return gelu_replacer(m, input);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithErfTwo::GeluFusionWithErfTwo() {
    MATCHER_SCOPE(GeluFusionWithErfTwo);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = pass::pattern::any_input();
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT2, 0.001f));
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({input, div_constant});

    // In case of ConvertDivideWithConstant is applied and Div is converted to Mul
    auto mul_as_div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT1_2, 0.001f));
    auto mul_as_div = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_as_div_constant});
    auto erf_input = std::make_shared<Or>(ov::OutputVector{div, mul_as_div});

    auto erf = ov::pass::pattern::wrap_type<ov::op::v0::Erf>({erf_input});
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(1.0f));
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({add_constant, erf});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(0.5f));

    // 0.5 * (x * (1 + erf(x / sqrt(2)))
    auto mul_first = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, add});
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({mul_constant, mul_first});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        return gelu_replacer(m, input);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithErfThree::GeluFusionWithErfThree() {
    MATCHER_SCOPE(GeluFusionWithErfThree);
    // Replaces a sub-graph with a Gelu op
    // Shared by every pattern: (1 + erf(x / sqrt(2)))
    auto input = pass::pattern::any_input();
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT2, 0.001f));
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({input, div_constant});

    // In case of ConvertDivideWithConstant is applied and Div is converted to Mul
    auto mul_as_div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT1_2, 0.001f));
    auto mul_as_div = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_as_div_constant});
    auto erf_input = std::make_shared<Or>(ov::OutputVector{div, mul_as_div});

    auto erf = ov::pass::pattern::wrap_type<ov::op::v0::Erf>({erf_input});
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(1.0f));
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({add_constant, erf});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(0.5f));

    // x * (0.5 * (1 + erf(x / sqrt(2)))
    auto mul_first = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add, mul_constant});
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_first});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        return gelu_replacer(m, input);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithErfFour::GeluFusionWithErfFour() {
    MATCHER_SCOPE(GeluFusionWithErfFour);
    using namespace ov;
    using namespace ov::pass::pattern;

    auto input = any_input();
    auto mul1_constant = wrap_type<ov::op::v0::Constant>(check_value(SQRT1_2, 0.001f));
    auto mul1 = wrap_type<ov::op::v1::Multiply>({input, mul1_constant});
    auto erf = wrap_type<ov::op::v0::Erf>({mul1});
    auto mul2_constant = wrap_type<ov::op::v0::Constant>(check_value(0.5f));
    auto mul2 = wrap_type<ov::op::v1::Multiply>({erf, mul2_constant});
    auto add_constant = wrap_type<ov::op::v0::Constant>(check_value(0.5f));
    auto add = wrap_type<ov::op::v1::Add>({add_constant, mul2});

    // x * (0.5 + 0.5 * erf(x * (1 / sqrt(2))))
    auto mul3 = wrap_type<ov::op::v1::Multiply>({input, add});

    matcher_pass_callback callback = [=](Matcher& m) {
        return gelu_replacer(m, input);
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
    auto pow_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(3.0f));
    auto pow = ov::pass::pattern::wrap_type<ov::op::v1::Power>({input, pow_constant});

    auto mul_0_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(0.044715f, 0.001f));
    auto mul_0 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({pow, mul_0_constant});

    auto add_0 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, mul_0});

    auto mul_1_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT_2_PI, 0.01f));
    auto mul_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_0, mul_1_constant});

    auto tanh = ov::pass::pattern::wrap_type<ov::op::v0::Tanh>({mul_1});

    auto add_1_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(1.0f));
    auto add_1 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({tanh, add_1_constant});

    auto mul_2_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(0.5f));

    // x * (0.5 * (1 + tanh))
    auto mul_2_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_1, mul_2_constant});
    auto mul_3_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_2_1});

    // (x * 0.5) * (1 + tanh)
    auto mul_2_2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_2_constant});
    auto mul_3_2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_1, mul_2_2});

    auto mul_3 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_3_1, mul_3_2});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        return gelu_replacer(m, input, op::GeluApproximationMode::TANH);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_3, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithTanhNoPower::GeluFusionWithTanhNoPower() {
    // Replaces a sub-graph with a Gelu (ov::op::v0::Tanh) op
    // x * 0.5 * (1 + tanh((x * 0.044715 * x + 1) * x * sqrt(2 / pi)))
    MATCHER_SCOPE(GeluFusionWithTanhNoPower);
    auto input = pattern::any_input();

    auto const1 = pattern::wrap_type<ov::op::v0::Constant>(check_value(0.044715f, 0.001f));
    auto mul1 = pattern::wrap_type<ov::op::v1::Multiply>({input, const1});

    auto mul2 = pattern::wrap_type<ov::op::v1::Multiply>({mul1, input});

    auto const2 = pattern::wrap_type<ov::op::v0::Constant>(check_value(1.0f));
    auto add1 = pattern::wrap_type<ov::op::v1::Add>({const2, mul2});

    auto const3 = pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT_2_PI, 0.01f));
    auto mul3 = pattern::wrap_type<ov::op::v1::Multiply>({input, const3});

    auto mul4 = pattern::wrap_type<ov::op::v1::Multiply>({add1, mul3});

    auto tanh = pattern::wrap_type<ov::op::v0::Tanh>({mul4});

    auto const4 = pattern::wrap_type<ov::op::v0::Constant>(check_value(1.0f));
    auto add2 = pattern::wrap_type<ov::op::v1::Add>({tanh, const4});

    auto const5 = pattern::wrap_type<ov::op::v0::Constant>(check_value(0.5f));
    auto mul5 = pattern::wrap_type<ov::op::v1::Multiply>({input, const5});

    auto mul6 = pattern::wrap_type<ov::op::v1::Multiply>({add2, mul5});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        return gelu_replacer(m, input, op::GeluApproximationMode::TANH);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul6, matcher_name);
    register_matcher(m, callback);
}

ov::pass::GeluFusionWithTanhNoPower2::GeluFusionWithTanhNoPower2() {
    MATCHER_SCOPE(GeluFusionWithTanhNoPower2);
    // Replaces a sub-graph with a Gelu (ov::op::v0::Tanh) op
    // x * (0.5 * (1 + tanh([sqrt(2 / pi)] * [x + 0.044715 * x * x * x])))
    auto input = pass::pattern::any_input();

    auto mul_0 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, input});
    auto mul_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_0});

    auto mul_2_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(0.044715f, 0.001f));
    auto mul_2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({mul_1, mul_2_constant});

    auto add_0 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, mul_2});

    auto mul_3_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(SQRT_2_PI, 0.01f));
    auto mul_3 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_0, mul_3_constant});

    auto tanh = ov::pass::pattern::wrap_type<ov::op::v0::Tanh>({mul_3});

    auto add_1_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(1.0f));
    auto add_1 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({tanh, add_1_constant});

    auto mul_4_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(check_value(0.5f));

    // x * (0.5 * (1 + tanh))
    auto mul_4_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_1, mul_4_constant});
    auto mul_5_1 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_4_1});

    // (x * 0.5) * (1 + tanh)
    auto mul_4_2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_4_constant});
    auto mul_5_2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_1, mul_4_2});

    // 0.5 * (x * (1 + tanh))
    auto mul_4_3 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({add_1, input});
    auto mul_5_3 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({mul_4_3, mul_4_constant});

    auto mul_5 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_5_1, mul_5_2, mul_5_3});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        return gelu_replacer(m, input, op::GeluApproximationMode::TANH);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_5, matcher_name);
    register_matcher(m, callback);
}
