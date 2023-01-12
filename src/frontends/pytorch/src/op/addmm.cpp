// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_addmm(NodeContext& context) {
    auto input = context.get_input(0);
    auto m1 = context.get_input(1);
    auto m2 = context.get_input(2);
    auto beta = context.get_input(3);
    auto alpha = context.get_input(4);
    auto beta_converted = context.mark_node(std::make_shared<opset8::ConvertLike>(beta, input));
    auto mm = context.mark_node(std::make_shared<opset8::MatMul>(m1, m2));
    auto alpha_converted = context.mark_node(std::make_shared<opset8::ConvertLike>(alpha, mm));
    auto input_beta = context.mark_node(std::make_shared<opset8::Multiply>(input, beta_converted));
    auto mm_alpha = context.mark_node(std::make_shared<opset8::Multiply>(mm, alpha_converted));
    return {context.mark_node(std::make_shared<opset8::Add>(input_beta, mm_alpha))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
