// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_rad2deg(const NodeContext& context) {
    // Ensure that the operation has exactly one input (the input tensor)
    num_inputs_check(context, 1, 1);

    // Retrieve the input tensor
    auto input = context.get_input(0);

    // Get the input element type dynamically
    auto input_type = input.get_element_type();

    const double pi_val = std::atan(1.0) * 4;

    // Create a constant node with the conversion factor (180 / π) using fp64 type
    auto conversion_factor_fp64 =
        context.mark_node(ov::op::v0::Constant::create(ov::element::f64, Shape{}, {180.0 / pi_val}));

    // Convert the constant to the same type as the input tensor using ConvertLike
    auto conversion_factor =
        context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(conversion_factor_fp64, input));

    // Apply the multiplication operation to convert radians to degrees
    auto result = context.mark_node(std::make_shared<ov::op::v1::Multiply>(input, conversion_factor));

    // Return the computed result as an OutputVector
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov