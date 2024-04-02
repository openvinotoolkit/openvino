// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rrelu_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    float default_lower = 1/8.0;
    float default_upper=1/3.0;
    Output<Node> lower = ov::op::v0::Constant::create(element::f32, Shape{1}, {default_lower});
    Output<Node> upper = ov::op::v0::Constant::create(element::f32, Shape{1}, {default_upper});
    if (context.get_input_size() == 1) {
       lower = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(lower, x));
       upper = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(upper, x));
    } else {
       lower = context.get_input(1);
       upper = context.get_input(2);
    }
    Output<Node> lower_plus_upper = std::make_shared<ov::op::v1::Add>(lower, upper);
    Output<Node> average = std::make_shared<ov::op::v1::Divide>(lower_plus_upper, ov::op::v0::Constant::create(element::f32, Shape{1}, {2.0f}));
    average = context.mark_node(average);
    return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};     
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
