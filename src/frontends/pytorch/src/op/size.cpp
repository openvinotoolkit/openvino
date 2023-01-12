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

OutputVector translate_size(NodeContext& context) {
    auto shape = context.mark_node(std::make_shared<opset8::ShapeOf>(context.get_input(0), element::i32));
    if (context.input_is_none(1)) {
        return shape->outputs();
    } else {
        auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
        return {context.mark_node(std::make_shared<opset8::Gather>(shape, context.get_input(1), axis_0))};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
