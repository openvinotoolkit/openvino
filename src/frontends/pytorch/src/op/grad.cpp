// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_grad(const NodeContext& context) {
    // aten::grad is used for gradient computation during training.
    // In inference mode, gradients are not needed, so we return zeros with the same shape as input.
    // Schema: aten::grad(Tensor outputs, Tensor inputs, ...) -> Tensor
    num_inputs_check(context, 1, 10);
    
    // Get the first input (outputs tensor) to determine the shape
    auto input = context.get_input(0);
    
    // Create zero constant
    auto zero_value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    
    // Get shape of input
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    
    // Broadcast zero to match input shape
    auto zeros = context.mark_node(std::make_shared<v3::Broadcast>(zero_value, input_shape));
    
    // Convert to match input dtype
    auto result = context.mark_node(std::make_shared<v1::ConvertLike>(zeros, input));
    
    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
