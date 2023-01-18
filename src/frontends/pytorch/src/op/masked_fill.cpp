// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_masked_fill(NodeContext& context) {
    auto data = context.get_input(0);
    auto mask = context.get_input(1);
    auto value = context.const_input<float>(2);
    auto data_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(data));
    auto value_const = context.mark_node(opset10::Constant::create(element::f32, Shape({}), {value}));
    auto broadcasted_value = context.mark_node(std::make_shared<opset10::Broadcast>(value_const, data_shape));
    auto bool_mask = context.mark_node(std::make_shared<opset10::Convert>(mask, element::boolean));
    return {context.mark_node(std::make_shared<opset10::Select>(bool_mask, broadcasted_value, data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov