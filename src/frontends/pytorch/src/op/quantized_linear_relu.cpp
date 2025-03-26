// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/relu.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantized_relu(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    
    const auto x = context.get_input(0);
    const auto scale = context.get_input(1);
    const auto zero_point = context.get_input(2);

    // Step 1: Apply ReLU activation
    const auto quantized_relu = context.mark_node(std::make_shared<v0::Relu>(x));

    // Step 2: Requantize the output
    return {quantize(context, quantized_relu, scale, zero_point, x)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
