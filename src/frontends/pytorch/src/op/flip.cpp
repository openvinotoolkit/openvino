// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_flip(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto axis = context.get_input(1);
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto minimum_int =
        context.mark_node(v0::Constant::create(element::i32, Shape{}, {std::numeric_limits<int>::min()}));
    auto axis_shape = context.mark_node(std::make_shared<v3::ShapeOf>(axis, element::i32));
    auto start = context.mark_node(std::make_shared<v3::Broadcast>(minus_one, axis_shape));
    auto stop = context.mark_node(std::make_shared<v3::Broadcast>(minimum_int, axis_shape));
    auto slice = context.mark_node(std::make_shared<v8::Slice>(x, start, stop, start, axis));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, slice);
    }
    return {slice};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov