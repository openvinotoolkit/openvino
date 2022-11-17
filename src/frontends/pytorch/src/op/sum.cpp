// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_sum(NodeContext& context) {
    bool keep_dims = false;
    ov::Output<ov::Node> axes;
    Output<Node> cast;
    FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() >= 1, "Operation has no inputs.");
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0), "Input should not be None.");
    auto data = context.get_input(0);
    if (context.input_is_none(1)) {
        auto start = std::make_shared<opset8::Constant>(element::i32, Shape{}, 0);
        auto step = std::make_shared<opset8::Constant>(element::i32, Shape{}, 1);
        auto shape = context.mark_node(std::make_shared<opset8::ShapeOf>(data, element::i32));
        auto rank = context.mark_node(std::make_shared<opset8::ShapeOf>(shape, element::i32));
        auto reduced_rank = context.mark_node(std::make_shared<opset8::Squeeze>(rank));
        axes = context.mark_node(std::make_shared<opset8::Range>(start, reduced_rank, step, element::i32));
    } else {
        axes = context.get_input(1);
    }
    if (context.get_input_size() >= 2 && !context.input_is_none(2)) {
        keep_dims = context.const_input<bool>(2);
    }

    return {context.mark_node(std::make_shared<opset8::ReduceSum>(data, axes, keep_dims))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov