// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_norm(NodeContext& context) {
    auto input_tensor = context.get_input(0);
    auto p = context.const_input<float>(1);
    auto dim = context.get_input(2);
    auto keep_dim = context.const_input<bool>(3);

    OutputVector res;

    if (p == 1) {
        auto reduce_l1 = context.mark_node(std::make_shared<opset10::ReduceL1>(input_tensor, dim, keep_dim));
        res.push_back(reduce_l1);
    } else if (p == 2) {
        auto reduce_l2 = context.mark_node(std::make_shared<opset10::ReduceL2>(input_tensor, dim, keep_dim));
        res.push_back(reduce_l2);
    } else if (p == std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<opset10::Abs>(input_tensor));
        auto max = context.mark_node(std::make_shared<opset10::ReduceMax>(abs, dim, keep_dim));
        res.push_back(max);
    } else if (p == -std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<opset10::Abs>(input_tensor));
        auto min = context.mark_node(std::make_shared<opset10::ReduceMin>(abs, dim, keep_dim));
        res.push_back(min);
    } else {
        auto const_p = context.mark_node(opset10::Constant::create(element::f64, Shape{1}, {p}));
        auto const_p_inv = context.mark_node(opset10::Constant::create(element::f64, Shape{1}, {1.0 / p}));
        auto abs = context.mark_node(std::make_shared<opset10::Abs>(input_tensor));
        auto pow = context.mark_node(std::make_shared<opset10::Power>(abs, const_p));
        auto sum = context.mark_node(std::make_shared<opset10::ReduceSum>(pow, dim, keep_dim));
        auto pow_inv = context.mark_node(std::make_shared<opset10::Power>(sum, const_p_inv));
        res.push_back(pow_inv);
    }

    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov