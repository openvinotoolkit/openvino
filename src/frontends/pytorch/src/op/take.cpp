// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_take(const NodeContext& node) {
    num_inputs_check(node, 2, 2);
    auto input = node.get_input(0);
    auto index = node.get_input(1);
    
    // Flatten input to 1D
    auto reshape_pattern = node.mark_node(opset10::Constant::create(element::i32, Shape{1}, {-1}));
    auto flattened = node.mark_node(std::make_shared<opset10::Reshape>(input, reshape_pattern, false));
    
    // Gather
    // axis is 0 since it is flat
    auto axis = node.mark_node(opset10::Constant::create(element::i32, Shape{}, {0}));
    auto gather = node.mark_node(std::make_shared<opset10::Gather>(flattened, index, axis));
    
    return {gather};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
