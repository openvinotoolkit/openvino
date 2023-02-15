// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_embedding(NodeContext& context) {
    num_inputs_check(context, 5, 5);
    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    // TODO: find out the meaning of input idx 2
    FRONT_END_OP_CONVERSION_CHECK(
        context.const_input<bool>(3) == false && context.const_input<bool>(4) == false,
        "Only False is supported on inputs with indexes 3 and 4 for aten::embedding translation");
    auto axis_0 = context.mark_node(ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
    return {context.mark_node(std::make_shared<ov::op::v8::Gather>(data, indices, axis_0))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov