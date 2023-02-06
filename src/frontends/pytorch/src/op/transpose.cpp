// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_transpose(NodeContext& context) {
    auto dim0 = context.const_input<int64_t>(1);
    auto dim1 = context.const_input<int64_t>(2);
    auto shape = std::make_shared<opset10::ShapeOf>(context.get_input(0), element::i32);
    auto rank_ = std::make_shared<opset10::ShapeOf>(shape, element::i32);
    auto rank = std::make_shared<opset10::Squeeze>(rank_);
    // Use opset::If for dim normalization
    auto dim0_node = context.get_input(1);
    auto dim1_node = context.get_input(2);
    if (dim0 < 0) {
        dim0_node = std::make_shared<opset10::Add>(rank, dim0_node);
    }
    if (dim1 < 0) {
        dim1_node = std::make_shared<opset10::Add>(rank, dim1_node);
    }
    auto start = opset10::Constant::create(element::i32, {}, {0});
    auto step = opset10::Constant::create(element::i32, {}, {1});
    auto range = std::make_shared<opset10::Range>(start, rank, step, element::i32);

    auto axis_0 = opset10::Constant::create(element::i64, Shape{}, {0});
    auto dim0_node_ = std::make_shared<opset10::Unsqueeze>(dim0_node, axis_0);
    auto dim1_node_ = std::make_shared<opset10::Unsqueeze>(dim1_node, axis_0);
    auto indices = std::make_shared<opset10::Concat>(OutputVector{dim0_node_, dim1_node_}, 0);
    auto updates = std::make_shared<opset10::Concat>(OutputVector{dim1_node_, dim0_node_}, 0);
    auto scatter = std::make_shared<opset10::ScatterElementsUpdate>(range, indices, updates, axis_0);
    context.mark_nodes(
        {shape, rank_, rank, start, step, range, axis_0, dim0_node_, dim1_node_, indices, updates, scatter});

    return {context.mark_node(std::make_shared<opset10::Transpose>(context.get_input(0), scatter))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov