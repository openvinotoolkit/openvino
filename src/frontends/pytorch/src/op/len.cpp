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

OutputVector translate_len(NodeContext& context) {
    auto const_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}));
    auto const_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));
    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input, element::i64));

    auto slice = context.mark_node(std::make_shared<opset8::Slice>(input_shape, const_0, const_1, const_1));
    auto squeeze = std::make_shared<opset8::Squeeze>(slice, const_0);
    return {context.mark_node(squeeze)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov