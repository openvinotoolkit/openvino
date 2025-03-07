// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rrelu(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);

    float lower_default = 1.0f / 8.0f;
    float upper_default = 1.0f / 3.0f;

    Output<Node> lower = ov::op::v0::Constant::create(element::f32, Shape{1}, {lower_default});
    Output<Node> upper = ov::op::v0::Constant::create(element::f32, Shape{1}, {upper_default});

    if (context.get_input_size() > 1) {
        lower = context.get_input(1);
        upper = context.get_input(2);
    }
    lower = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(lower, x));
    upper = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(upper, x));

    Output<Node> negative_slope = context.mark_node(std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::Add>(lower, upper),
        ov::op::v0::Constant::create(element::f32, Shape{1}, {2.0f})
    ));

    return {context.mark_node(std::make_shared<v0::PRelu>(x, negative_slope))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
