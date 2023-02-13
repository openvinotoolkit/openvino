// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_len(NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto const_0 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i64));

    auto slice = context.mark_node(std::make_shared<v8::Slice>(input_shape, const_0, const_1, const_1));
    auto squeeze = std::make_shared<v0::Squeeze>(slice, const_0);
    return {context.mark_node(squeeze)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov