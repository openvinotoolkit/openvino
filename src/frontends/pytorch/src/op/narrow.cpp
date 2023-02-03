// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_narrow(NodeContext& context) {
    auto input_tensor = context.get_input(0);
    auto axis_const = context.const_input<int64_t>(1);
    auto start_const = context.const_input<int64_t>(2);
    auto length_const = context.const_input<int64_t>(3);
    auto axis = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {axis_const}));
    auto start = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {start_const}));
    auto length = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {length_const}));
    auto const_one = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {1}));
    auto stop = context.mark_node(std::make_shared<opset10::Add>(start, length));

    auto narrow = context.mark_node(std::make_shared<opset10::Slice>(input_tensor, start, stop, const_one, axis));
    return {narrow};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
