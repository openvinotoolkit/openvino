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

OutputVector translate_repeat(NodeContext& context) {
    auto x = context.get_input(0);
    auto repeats = context.get_input(1);
    auto one = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
    auto sizes_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(repeats, element::i64));
    auto expand_shape = context.mark_node(std::make_shared<opset8::Broadcast>(one, sizes_shape));
    auto expanded_input =
        context.mark_node(std::make_shared<opset8::Broadcast>(x, expand_shape, ov::op::BroadcastType::BIDIRECTIONAL));
    return {context.mark_node(std::make_shared<opset8::Tile>(expanded_input, repeats))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov