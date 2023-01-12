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

OutputVector translate_rsqrt(NodeContext& context) {
    auto data = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(data));
    auto one_const = context.mark_node(opset8::Constant::create(element::f32, Shape({}), {1}));
    auto sqrt_data = context.mark_node(std::make_shared<opset8::Sqrt>(data));
    return {context.mark_node(std::make_shared<opset8::Divide>(one_const, sqrt_data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov