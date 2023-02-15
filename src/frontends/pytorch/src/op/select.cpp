// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_select(NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_minus_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));

    auto input_tensor = context.get_input(0);
    auto dim = context.mark_node(std::make_shared<v1::Reshape>(context.get_input(1), const_1, false));
    auto start = context.mark_node(std::make_shared<v1::Reshape>(context.get_input(2), const_1, false));

    auto less = context.mark_node(std::make_shared<v1::Less>(start, const_0));
    auto const_1_signed = context.mark_node(std::make_shared<v1::Select>(less, const_minus_1, const_1));
    auto stop = context.mark_node(std::make_shared<v1::Add>(start, const_1_signed));

    auto slice_node = context.mark_node(std::make_shared<v8::Slice>(input_tensor, start, stop, const_1_signed, dim));

    return {context.mark_node(std::make_shared<v0::Squeeze>(slice_node, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
