// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"
#include <cmath>  // Required for M_PI

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_rad2deg(const NodeContext& context) {
    // Ensure that the operation has exactly one input (the input tensor)
    num_inputs_check(context, 1, 1);
    
    // Retrieve the input tensor
    auto input = context.get_input(0);
    
    // Create a constant node with the conversion factor (180 / π)
    auto conversion_factor = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape{}, {180.0 / M_PI}));
    
    // Apply the multiplication operation to convert radians to degrees
    auto result = context.mark_node(std::make_shared<ov::op::v1::Multiply>(input, conversion_factor));

    // Return the computed result as an OutputVector
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
