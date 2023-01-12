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

OutputVector translate_layer_norm(NodeContext& context) {
    auto eps = context.const_input<float>(4);
    auto normalized_shape = context.const_input<Shape>(1);
    FRONT_END_OP_CONVERSION_CHECK(normalized_shape.size() == 1,
                                  "Translation for aten::layer_norm supports only single normalized_shape value, "
                                  "which means normalizing over the last dimension.");
    // TODO: support any dimention
    auto axes = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1}));
    auto out_node = context.mark_node(
        std::make_shared<opset8::MVN>(context.get_input(0), axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT));
    if (!context.input_is_none(2)) {
        out_node = context.mark_node(std::make_shared<opset8::Multiply>(out_node, context.get_input(2)));
    }
    if (!context.input_is_none(3)) {
        out_node = context.mark_node(std::make_shared<opset8::Add>(out_node, context.get_input(3)));
    }
    return {out_node};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov