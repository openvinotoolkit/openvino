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

OutputVector translate_select(NodeContext& context) {
    auto const_1 = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {1}));
    auto const_minus_1 = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {-1}));
    auto const_0 = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {0}));
    auto input_tensor = context.get_input(0);
    auto dim = context.mark_node(std::make_shared<opset10::Reshape>(context.get_input(1), const_1, false));
    auto start = context.mark_node(std::make_shared<opset10::Reshape>(context.get_input(2), const_1, false));

    auto less = context.mark_node(std::make_shared<opset10::Less>(start, const_0));
    auto const_1_signed = context.mark_node(std::make_shared<opset10::Select>(less, const_minus_1, const_1));
    auto stop = context.mark_node(std::make_shared<opset10::Add>(start, const_1_signed));

    auto slice_node =
        context.mark_node(std::make_shared<opset10::Slice>(input_tensor, start, stop, const_1_signed, dim));

    return {context.mark_node(std::make_shared<opset10::Squeeze>(slice_node, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
