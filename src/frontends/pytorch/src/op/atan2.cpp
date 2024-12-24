// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include <math.h>

#include <memory>

#include "openvino/core/type/element_type.hpp"
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
#include "openvino/op/logical_or.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_atan2(const NodeContext& context) {
    // atan2(input, other, *) → Tensor
    num_inputs_check(context, 2, 2);
    Output<Node> lhs;
    Output<Node> rhs;

    std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);

    auto div = context.mark_node(std::make_shared<v1::Divide>(lhs, rhs));

    auto atan = context.mark_node(std::make_shared<v0::Atan>(div));

    // create some constants to adjust result according to quadrant.
    auto zero = context.mark_node(v0::Constant::create(ov::element::i32, Shape{}, {0}));
    auto pi = context.mark_node(v0::Constant::create(ov::element::f64, Shape{}, {M_PI}));
    auto half_pi = context.mark_node(v0::Constant::create(ov::element::f64, Shape{}, {M_PI_2}));
    auto neg_half_pi = context.mark_node(v0::Constant::create(ov::element::f64, Shape{}, {-M_PI_2}));

    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, rhs));
    pi = context.mark_node(std::make_shared<v1::ConvertLike>(pi, rhs));
    half_pi = context.mark_node(std::make_shared<v1::ConvertLike>(half_pi, rhs));
    neg_half_pi = context.mark_node(std::make_shared<v1::ConvertLike>(neg_half_pi, rhs));

    //  x > 0, no adjustment needed
    auto x_greater_than_zero = context.mark_node(std::make_shared<v1::Greater>(rhs, zero));

    // x < 0 and y >= 0, need to plus pi
    auto y_greater_equal_zero = context.mark_node(std::make_shared<v1::GreaterEqual>(lhs, zero));
    auto x_less_than_zero = context.mark_node(std::make_shared<v1::Less>(rhs, zero));
    auto add_pi_condition = context.mark_node(std::make_shared<v1::LogicalAnd>(x_less_than_zero, y_greater_equal_zero));

    // x < 0 and y < 0, need to minus pi
    auto y_less_than_zero = std::make_shared<v1::Less>(lhs, zero);
    auto subtract_pi_condition =
        context.mark_node(std::make_shared<v1::LogicalAnd>(x_less_than_zero, y_less_than_zero));

    // x = 0 and y > 0, pi/2
    auto x_equal_zero = std::make_shared<v1::Equal>(rhs, zero);
    auto y_greater_than_zero = std::make_shared<v1::Greater>(lhs, zero);
    auto half_pi_condition = context.mark_node(std::make_shared<v1::LogicalAnd>(x_equal_zero, y_greater_than_zero));

    // x = 0 and y < 0, -pi/2
    auto neg_half_pi_condition = context.mark_node(std::make_shared<v1::LogicalAnd>(x_equal_zero, y_less_than_zero));

    auto special_case_condition =
        context.mark_node(std::make_shared<v1::LogicalOr>(half_pi_condition, neg_half_pi_condition));

    // do adjustment
    auto atan_plus_pi = context.mark_node(std::make_shared<v1::Add>(atan, pi));
    auto atan_minus_pi = context.mark_node(std::make_shared<v1::Subtract>(atan, pi));

    // select result
    auto ajusted_case = context.mark_node(std::make_shared<v1::Select>(add_pi_condition, atan_plus_pi, atan_minus_pi));
    auto special_case = context.mark_node(std::make_shared<v1::Select>(half_pi_condition, half_pi, neg_half_pi));
    auto adjusted_atan = context.mark_node(std::make_shared<v1::Select>(x_greater_than_zero, atan, ajusted_case));
    auto result = context.mark_node(std::make_shared<v1::Select>(special_case_condition, special_case, adjusted_atan));

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
