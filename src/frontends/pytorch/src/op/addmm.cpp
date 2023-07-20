// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_addmm(const NodeContext& context) {
    num_inputs_check(context, 3, 5);
    auto input = context.get_input(0);
    auto m1 = context.get_input(1);
    auto m2 = context.get_input(2);
    auto mm = context.mark_node(std::make_shared<v0::MatMul>(m1, m2));
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    ov::Output<Node> alpha = one;
    ov::Output<Node> beta = one;
    if (!context.input_is_none(3)) {
        beta = context.get_input(3);
    }
    if (!context.input_is_none(4)) {
        alpha = context.get_input(4);
    }
    auto beta_converted = context.mark_node(std::make_shared<v1::ConvertLike>(beta, input));
    auto alpha_converted = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, mm));
    auto input_beta = context.mark_node(std::make_shared<v1::Multiply>(input, beta_converted));
    auto mm_alpha = context.mark_node(std::make_shared<v1::Multiply>(mm, alpha_converted));
    return {context.mark_node(std::make_shared<v1::Add>(input_beta, mm_alpha))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
