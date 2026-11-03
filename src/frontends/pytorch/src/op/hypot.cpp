// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sqrt.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_hypot(const NodeContext& context) {
    // hypot(input, other, *) -> Tensor : elementwise sqrt(input^2 + other^2), with broadcasting.
    num_inputs_check(context, 2, 2);
    Output<Node> lhs;
    Output<Node> rhs;
    std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);

    // Squaring the inputs directly overflows (or underflows to zero) long before the result does,
    // so compute it as max * sqrt(1 + (min / max)^2), where min / max is in [0, 1].
    const auto lhs_abs = context.mark_node(std::make_shared<v0::Abs>(lhs));
    const auto rhs_abs = context.mark_node(std::make_shared<v0::Abs>(rhs));
    const auto max_abs = context.mark_node(std::make_shared<v1::Maximum>(lhs_abs, rhs_abs));
    const auto min_abs = context.mark_node(std::make_shared<v1::Minimum>(lhs_abs, rhs_abs));

    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, lhs));
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, lhs));
    auto inf = context.mark_node(v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::infinity()}));
    inf = context.mark_node(std::make_shared<v1::ConvertLike>(inf, lhs));
    auto nan =
        context.mark_node(v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::quiet_NaN()}));
    nan = context.mark_node(std::make_shared<v1::ConvertLike>(nan, lhs));

    // Both inputs zero: divide by 1 instead of 0, the result is then 0 * sqrt(1) = 0.
    const auto max_is_zero = context.mark_node(std::make_shared<v1::Equal>(max_abs, zero));
    const auto safe_max = context.mark_node(std::make_shared<v1::Select>(max_is_zero, one, max_abs));
    const auto ratio = context.mark_node(std::make_shared<v1::Divide>(min_abs, safe_max));
    const auto ratio_squared = context.mark_node(std::make_shared<v1::Multiply>(ratio, ratio));
    const auto one_plus = context.mark_node(std::make_shared<v1::Add>(one, ratio_squared));
    const auto root = context.mark_node(std::make_shared<v0::Sqrt>(one_plus));
    Output<Node> result = context.mark_node(std::make_shared<v1::Multiply>(max_abs, root));

    // Maximum/Minimum do not reliably propagate NaN, so restore it explicitly.
    const auto lhs_is_nan = context.mark_node(std::make_shared<v10::IsNaN>(lhs));
    const auto rhs_is_nan = context.mark_node(std::make_shared<v10::IsNaN>(rhs));
    const auto any_nan = context.mark_node(std::make_shared<v1::LogicalOr>(lhs_is_nan, rhs_is_nan));
    result = context.mark_node(std::make_shared<v1::Select>(any_nan, nan, result));

    // As in std::hypot, an infinite input gives +inf even when the other input is NaN.
    const auto lhs_is_inf = context.mark_node(std::make_shared<v10::IsInf>(lhs));
    const auto rhs_is_inf = context.mark_node(std::make_shared<v10::IsInf>(rhs));
    const auto any_inf = context.mark_node(std::make_shared<v1::LogicalOr>(lhs_is_inf, rhs_is_inf));
    result = context.mark_node(std::make_shared<v1::Select>(any_inf, inf, result));

    return {result};
}

}  // namespace ov::frontend::pytorch::op
