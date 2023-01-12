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

namespace {
OutputVector base_expand(NodeContext& context, ov::Output<ov::Node> x, ov::Output<ov::Node> sizes) {
    auto one = context.mark_node(opset8::Constant::create(element::i32, Shape{}, {1}));
    auto sizes_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(sizes, element::i32));
    auto neg_one = context.mark_node(opset8::Constant::create(element::i32, Shape{}, {-1}));
    auto neg_ones = context.mark_node(std::make_shared<opset8::Broadcast>(neg_one, sizes_shape));
    auto ones = context.mark_node(std::make_shared<opset8::Broadcast>(one, sizes_shape));
    auto neg_sizes = context.mark_node(std::make_shared<opset8::Equal>(sizes, neg_ones));
    auto shape = context.mark_node(std::make_shared<opset8::Select>(neg_sizes, ones, sizes));
    return {std::make_shared<opset8::Broadcast>(x, shape, ov::op::BroadcastType::BIDIRECTIONAL)};
};
}  // namespace

OutputVector translate_expand(NodeContext& context) {
    auto x = context.get_input(0);
    auto sizes = context.get_input(1);
    return base_expand(context, x, sizes);
};

OutputVector translate_expand_as(NodeContext& context) {
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<opset8::ShapeOf>(y, element::i32));
    return base_expand(context, x, sizes);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov