// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
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
    num_inputs_check(context, 2, 3);
    // "aten::atan2.out(Tensor input,Tensor other, *,Tensor(a!) out) → Tensor(a!)"
    Output<Node> y;
    Output<Node> x;
    std::tie(y, x) = get_inputs_with_promoted_types(context, 0, 1);
    auto dummy_const = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape({}), {0.5}))->output(0);
    align_eltwise_input_types(context, x, dummy_const, false, true);

    // handle the first condition : x>0
    align_eltwise_input_types(context, y, x, false, true);
    auto div_y_x = context.mark_node(std::make_shared<v1::Divide>(y, x));
    auto atan = context.mark_node(std::make_shared<v0::Atan>(div_y_x));
    auto const_zero = v0::Constant::create(element::f32, Shape{}, {0});
    auto result = atan->output(0);

    // handle the second condition : x<0 && y>=0
    auto const_pi = v0::Constant::create(element::f32, Shape{}, {std::atan(1.0) * 4});
    align_eltwise_input_types(context, x, const_pi, false, true);
    auto is_x_negative = context.mark_node(std::make_shared<v1::Less>(x, const_zero));
    align_eltwise_input_types(context, y, const_zero, false, true);
    auto y_non_negative = context.mark_node(std::make_shared<v1::GreaterEqual>(y, const_zero));
    auto cond1 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_negative, y_non_negative));
    align_eltwise_input_types(context, atan, const_pi, false, true);
    auto atan_y_x_plus_pi = context.mark_node(std::make_shared<v1::Add>(atan, const_pi));
    result = context.mark_node(std::make_shared<v1::Select>(cond1, atan_y_x_plus_pi, result));

    // handle the third condition : x<0 && y<0
    align_eltwise_input_types(context, x, const_zero, false, true);
    auto is_y_negative = context.mark_node(std::make_shared<v1::Less>(y, const_zero));
    auto cond2 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_negative, is_y_negative));
    align_eltwise_input_types(context, atan, const_pi, false, true);
    auto atan_y_x_minus_pi = context.mark_node(std::make_shared<v1::Subtract>(atan, const_pi));
    result = context.mark_node(std::make_shared<v1::Select>(cond2, atan_y_x_minus_pi, result));

    // handle the fourth condition : x=0 && y>0
    align_eltwise_input_types(context, x, const_zero, false, true);
    auto is_x_zero = context.mark_node(std::make_shared<v1::Equal>(x, const_zero));
    align_eltwise_input_types(context, y, const_zero, false, true);
    auto is_y_positive = context.mark_node(std::make_shared<v1::Greater>(y, const_zero));
    auto cond3 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_zero, is_y_positive));
    auto const_two = v0::Constant::create(element::f32, Shape{}, {2});
    align_eltwise_input_types(context, const_pi, const_two, false, true);
    auto pi_div_two = context.mark_node(std::make_shared<v1::Divide>(const_pi, const_two));
    result = context.mark_node(std::make_shared<v1::Select>(cond3, pi_div_two, result));

    // handle the fifth condition : x=0 && y<0
    auto cond4 = context.mark_node(std::make_shared<v1::LogicalAnd>(is_x_zero, is_y_negative));
    auto const_minus_two = v0::Constant::create(element::f32, Shape{}, {-2});
    align_eltwise_input_types(context, const_pi, const_minus_two, false, true);
    auto pi_div_minus_two = context.mark_node(std::make_shared<v1::Divide>(const_pi, const_minus_two));
    result = context.mark_node(std::make_shared<v1::Select>(cond4, pi_div_two, result));

    // check whether out tensor is given
    if(!context.input_is_none(2) && context.get_input_size() == 3) {
        auto out_tensor = context.get_input(2);
        // dtype is inherited from out tensor in input
    auto result_out = context.mark_node(std::make_shared<v1::ConvertLike>(result, out_tensor));

    return {result_out};
    }

    // when out tensor is not in input
    return {result};
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
