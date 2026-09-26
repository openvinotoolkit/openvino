// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_narrow(const NodeContext& context) {
    num_inputs_check(context, 4, 4);

    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto input_tensor = context.get_input(0);
    auto axis_input = context.get_input(1);
    auto start_input = context.get_input(2);
    auto length = context.get_input(3);

    Output<Node> start = context.mark_node(std::make_shared<v0::Unsqueeze>(start_input, const_0));
    auto axis = context.mark_node(std::make_shared<v0::Unsqueeze>(axis_input, const_0));

    // A negative start counts from the end of the dimension. Slice would accept it as is, but the stop
    // must be computed from the non-negative position: otherwise start=-2, length=2 gives stop=0, an
    // empty slice instead of the last two elements.
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(input_tensor, element::i32));
    auto scalar_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    Output<Node> dim_size = context.mark_node(std::make_shared<v8::Gather>(shape, axis, scalar_0));
    dim_size = context.mark_node(std::make_shared<v1::ConvertLike>(dim_size, start));
    start = normalize_axis(context, start, dim_size);
    auto stop = context.mark_node(std::make_shared<v1::Add>(start, length));

    auto narrow = context.mark_node(std::make_shared<v8::Slice>(input_tensor, start, stop, const_1, axis));
    return {narrow};
};

}  // namespace ov::frontend::pytorch::op