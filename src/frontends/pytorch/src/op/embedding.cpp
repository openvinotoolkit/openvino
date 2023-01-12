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

OutputVector translate_embedding(NodeContext& context) {
    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    // TODO: find out the meaning of input idx 2
    OV_FRONTEND_REQUIRE(context.const_input<bool>(3) == false);
    OV_FRONTEND_REQUIRE(context.const_input<bool>(4) == false);
    auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
    return {context.mark_node(std::make_shared<opset8::Gather>(data, indices, axis_0))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov