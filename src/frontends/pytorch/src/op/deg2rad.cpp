// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_deg2rad(const NodeContext& context) {
    // Ensure that the operation has exactly one input (the input tensor)
    num_inputs_check(context, 1, 1);

    // Retrieve the input tensor
    auto input = context.get_input(0);

    // Define the value of π
    const double pi_val = std::atan(1.0) * 4;

    // Create a constant node with the conversion factor (π / 180) using fp32 type
    Output<Node> conversion_factor_output = context.mark_node(
        ov::op::v0::Constant::create(ov::element::f32, Shape{}, {static_cast<float>(pi_val) / 180.0f}));

    // Align elementwise input types to handle integer cases correctly
    align_eltwise_input_types(context, input, conversion_factor_output, is_python_scalar_input(context, 0), true);

    // Apply the multiplication operation to convert degrees to radians
    auto result = context.mark_node(std::make_shared<ov::op::v1::Multiply>(input, conversion_factor_output));

    // Return the computed result as an OutputVector
    return {result};
}

}  // namespace ov::frontend::pytorch::op
