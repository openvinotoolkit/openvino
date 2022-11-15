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

OutputVector translate_select(NodeContext& context) {
    auto const_1 = opset8::Constant::create(element::i32, Shape{1}, {1});
    auto const_minus_1 = opset8::Constant::create(element::i32, Shape{1}, {-1});
    auto const_0 = opset8::Constant::create(element::i32, Shape{1}, {0});
    auto input_tensor = context.get_input(0);
    auto dim = std::make_shared<opset8::Reshape>(context.get_input(1), const_1, false);
    auto start = std::make_shared<opset8::Reshape>(context.get_input(2), const_1, false);

    auto less = std::make_shared<opset8::Less>(start, const_0);
    auto const_1_signed = std::make_shared<opset8::Select>(less, const_minus_1, const_1);
    auto stop = std::make_shared<opset8::Add>(start, const_1_signed);

    auto slice_node = std::make_shared<opset8::Slice>(input_tensor, start, stop, const_1_signed, dim);
    context.mark_node(slice_node);

    return {context.mark_node(std::make_shared<opset8::Squeeze>(slice_node, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
