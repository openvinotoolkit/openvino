// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_floor_divide(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
    const auto& type = x.get_element_type();
    if (!(type.is_static() && type.is_integral() && type.is_signed())) {
        auto div = context.mark_node(std::make_shared<v1::Divide>(x, y, true));
        return {context.mark_node(std::make_shared<v0::Floor>(div))};
    }
    // The plugins truncate signed integer division toward zero regardless of m_pythondiv
    // (openvinotoolkit/openvino#38248), so the quotient is one too large when the operands have different
    // signs and the division is inexact. Correct that explicitly. As with the integer handling in div.cpp,
    // this needs the element type at conversion time; a dynamic type keeps the plain Floor(Divide) above.
    // Truncating division on purpose: the correction below assumes it, whatever the plugin does.
    auto quotient = context.mark_node(std::make_shared<v1::Divide>(x, y, false));
    auto zero = context.mark_node(v0::Constant::create(type, Shape{}, {0}));
    auto minus_one = context.mark_node(v0::Constant::create(type, Shape{}, {-1}));
    auto remainder = context.mark_node(std::make_shared<v1::Mod>(x, y));
    auto x_negative = context.mark_node(std::make_shared<v1::Less>(x, zero));
    auto y_negative = context.mark_node(std::make_shared<v1::Less>(y, zero));
    auto signs_differ = context.mark_node(std::make_shared<v1::LogicalXor>(x_negative, y_negative));
    auto inexact = context.mark_node(std::make_shared<v1::NotEqual>(remainder, zero));
    auto round_down = context.mark_node(std::make_shared<v1::LogicalAnd>(signs_differ, inexact));
    auto correction = context.mark_node(std::make_shared<v1::Select>(round_down, minus_one, zero));
    return {context.mark_node(std::make_shared<v1::Add>(quotient, correction))};
};

}  // namespace ov::frontend::pytorch::op
