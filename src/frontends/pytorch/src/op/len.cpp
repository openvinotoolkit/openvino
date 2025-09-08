// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_len(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i64));

    auto slice = context.mark_node(std::make_shared<v8::Slice>(input_shape, const_0, const_1, const_1));
    // Slice will return empty tensor for empty lists, we use the fact that ReduceSum(empty tensor) = 0
    return {context.mark_node(std::make_shared<v1::ReduceSum>(slice, const_0, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov