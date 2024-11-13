// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_leaky_relu_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0); //  The input tensor

    Output<Node> lower = ov::op::v0::Constant::create(element::f32, Shape{1}, {1.0f / 8.0f});
    Output<Node> upper = ov::op::v0::Constant::create(element::f32, Shape{1}, {1.0f / 3.0f});

    // Override lower and upper bounds if passed as inputs
    if (!context.input_is_none(1)) {
        lower = context.get_input(1);
    }

    if (!context.input_is_none(2)) {
        upper = context.get_input(2);
    }

    // Convert lower/upper bounds to the same shape as x
    lower = context.mark_node(std::make_shared<v1::ConvertLike>(lower, x));
    upper = context.mark_node(std::make_shared<v1::ConvertLike>(upper, x));

    // Obtain a*x, necessary for RRelu
    auto lower_plus_upper = context.mark_node(std::make_shared<v1::Add>(lower, upper));
    auto a = context.mark_node(std::make_shared<v1::Divide>(lower_plus_upper, 2));
    auto a_times_x = context.mark_node(std::make_shared<v1::Multiply>(a, x));

    return {context.mark_node(std::make_shared<v0::PRelu>(x, a_times_x))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov