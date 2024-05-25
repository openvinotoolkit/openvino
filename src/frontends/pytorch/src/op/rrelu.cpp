// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
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
    float default_lower = 1 / 8.0;
    float default_upper = 1 / 3.0;
    Output<Node> lower = v0::Constant::create(element::f32, Shape{1}, {default_lower});
    Output<Node> upper = v0::Constant::create(element::f32, Shape{1}, {default_upper});
    if (!context.input_is_none(1)) {
        lower = context.get_input(1);
    }
    if (!context.input_is_none(2)) {
        upper = context.get_input(2);
    }
    lower = context.mark_node(std::make_shared<v1::ConvertLike>(lower, x));
    upper = context.mark_node(std::make_shared<v1::ConvertLike>(upper, x));
    auto lower_plus_upper = context.mark_node(std::make_shared<v1::Add>(lower, upper));
    auto two = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2.0f});
    two = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(two, x));
    auto average = context.mark_node(std::make_shared<v1::Divide>(lower_plus_upper, two));
    return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
