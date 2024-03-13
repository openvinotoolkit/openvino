// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_atan2(const NodeContext& context) {
    num_inputs_check(context, 3, 2);
    auto y = context.get_input(0);
    auto x = context.get_input(1);

    // handle the first condition : x>0
    auto div_y_x = context.mark_node(std::make_shared<v1::Divide>(y, x));
    auto atan = context.mark_node(std::make_shared<v0::Atan>(div_y_x));
    auto const_zero = v0::Constant::create(element::f64, Shape{}, {0});
    auto result = atan->output(0);

    // handle the second condition : x<0 && y>=0
    auto const_pi = v0::Constant::create(element::f64, Shape{}, {std::atan(1.0)*4});
    auto is_x_negative = context.mark_node(std::make_shared<v1::Less>(x, const_zero));
    auto y_non_negative = context.mark_node(std::make_shared<v1::GreaterEqual>(y, const_zero));
    auto cond1 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_negative, y_non_negative));
    auto atan_y_x_plus_pi = context.mark_node(std::make_shared<v1::Add>(atan, const_pi));
    result = context.mark_node(std::make_shared<v1::Select>(cond1, atan_y_x_plus_pi, result));

    // handle the third condition : x<0 && y<0
    auto is_y_negative = context.mark_node(std::make_shared<v1::Less>(y, const_zero));
    auto cond2 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_negative, is_y_negative));
    auto atan_y_x_minus_pi = context.mark_node(std::make_shared<v1::Subtract>(atan, const_pi));
    result = context.mark_node(std::make_shared<v1::Select>(cond2, atan_y_x_minus_pi, result));

    // handle the fourth condition : x=0 && y>0
    auto is_x_zero = context.mark_node(std::make_shared<v1::Equal>(x, const_zero));
    auto is_y_positive = context.mark_node(std::make_shared<v1::Greater>(y, const_zero));
    auto cond3 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_zero, is_y_positive));
    auto const_two = v0::Constant::create(element::f64, Shape{}, {2});
    auto pi_div_two = context.mark_node(std::make_shared<v1::Divide>(const_pi, const_two));
    result = context.mark_node(std::make_shared<v1::Select>(cond3, pi_div_two, result));

    // handle the fifth condition : x=0 && y<0
    auto cond4 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_zero, is_y_negative));
    auto const_minus_two = v0::Constant::create(element::f64, Shape{}, {-2});
    auto pi_div_minus_two = context.mark_node(std::make_shared<v1::Divide>(const_pi, const_minus_two));
    result = context.mark_node(std::make_shared<v1::Select>(cond4, pi_div_two, result));

    auto result_conv = context.mark_node(std::make_shared<v0::Convert>(result, context.get_input(2));

    return {result_conv};
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
