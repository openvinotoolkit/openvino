// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_narrow(const NodeContext& context) {
    num_inputs_check(context, 4, 4);

    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto input_tensor = context.get_input(0);
    auto axis_input = context.get_input(1);
    auto start_input = context.get_input(2);
    auto length = context.get_input(3);

    auto start = context.mark_node(std::make_shared<v0::Unsqueeze>(start_input, const_0));
    auto stop = context.mark_node(std::make_shared<v1::Add>(start, length));
    auto axis = context.mark_node(std::make_shared<v0::Unsqueeze>(axis_input, const_0));

    auto narrow = context.mark_node(std::make_shared<v8::Slice>(input_tensor, start, stop, const_1, axis));
    return {narrow};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov