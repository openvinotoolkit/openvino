// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

OutputVector translate_deg2rad(const NodeContext& context) {
    // Ensure that the operation has exactly one input (the input tensor)
    num_inputs_check(context, 1, 1);

    // Retrieve input tensor, promoting integer inputs to default floating point
    auto input = get_input_with_floating_type(context, 0);

    // Define π with double precision and compute conversion factor (π / 180)
    const double pi_val = 3.14159265358979323846;
    auto conversion_factor =
        context.mark_node(ov::op::v0::Constant::create(ov::element::f64, Shape{}, {pi_val / 180.0}))->output(0);
    conversion_factor = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(conversion_factor, input));

    // Apply the multiplication operation to convert degrees to radians
    auto result = context.mark_node(std::make_shared<ov::op::v1::Multiply>(input, conversion_factor));

    // Return the computed result as an OutputVector
    return {result};
}

}  // namespace ov::frontend::pytorch::op
