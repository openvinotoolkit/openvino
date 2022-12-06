// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_full(NodeContext& context) {
    auto sizes = context.get_input(0);
    auto value = context.get_input(1);
    return {context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes))};
};

OutputVector translate_full_like(NodeContext& context) {
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, input_shape));
    return {filled_tensor};
};

OutputVector translate_new_full(NodeContext& context) {
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.get_input(2);
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, input))};
};

OutputVector translate_zeros(NodeContext& context) {
    auto sizes = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    return {context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes))};
};

OutputVector translate_zeros_like(NodeContext& context) {
    auto input = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, input_shape));
    return {filled_tensor};
};

OutputVector translate_new_zeros(NodeContext& context) {
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, input))};
};

OutputVector translate_ones(NodeContext& context) {
    auto sizes = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {1}));
    return {context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes))};
};

OutputVector translate_ones_like(NodeContext& context) {
    auto input = context.get_input(0);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {1}));
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, input_shape));
    return {filled_tensor};
};

OutputVector translate_new_ones(NodeContext& context) {
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {1}));
    auto filled_tensor = context.mark_node(std::make_shared<opset8::Broadcast>(value, sizes));
    return {context.mark_node(std::make_shared<opset8::ConvertLike>(filled_tensor, input))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov