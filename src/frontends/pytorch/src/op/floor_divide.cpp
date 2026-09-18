// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_floor_divide(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
    const auto& type = x.get_element_type();
    // Floor(Divide) is exact for floating point and a no-op on integers, but the plugins truncate
    // integer division toward zero regardless of m_pythondiv (openvinotoolkit/openvino#38248), so
    // for signed integers the quotient is one too large when the operands have different signs and
    // the division is inexact. Correct that explicitly; unsigned integers never need it.
    // Truncating division on purpose: the correction below assumes it, whatever the plugin does.
    auto quotient = context.mark_node(std::make_shared<v1::Divide>(x, y, false));
    auto floored = context.mark_node(std::make_shared<v0::Floor>(quotient));
    if (type.is_static() && !(type.is_integral() && type.is_signed())) {
        return {floored};
    }
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, x));
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    minus_one = context.mark_node(std::make_shared<v1::ConvertLike>(minus_one, x));
    auto remainder = context.mark_node(std::make_shared<v1::Mod>(x, y));
    auto x_negative = context.mark_node(std::make_shared<v1::Less>(x, zero));
    auto y_negative = context.mark_node(std::make_shared<v1::Less>(y, zero));
    auto signs_differ = context.mark_node(std::make_shared<v1::LogicalXor>(x_negative, y_negative));
    auto inexact = context.mark_node(std::make_shared<v1::NotEqual>(remainder, zero));
    Output<Node> round_down = context.mark_node(std::make_shared<v1::LogicalAnd>(signs_differ, inexact));
    if (type.is_dynamic()) {
        // The element type is only known at compile time. 0.5 converts to 0 in any integer type and
        // stays 0.5 in a floating point one, which tells the two apart once the type is resolved and
        // folds away, leaving floating point inputs with plain Floor(Divide).
        auto half = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.5f}));
        half = context.mark_node(std::make_shared<v1::ConvertLike>(half, x));
        auto is_integral = context.mark_node(std::make_shared<v1::Equal>(half, zero));
        round_down = context.mark_node(std::make_shared<v1::LogicalAnd>(round_down, is_integral));
    }
    auto correction = context.mark_node(std::make_shared<v1::Select>(round_down, minus_one, zero));
    return {context.mark_node(std::make_shared<v1::Add>(floored, correction))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
