// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantize_per_tensor(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto input = context.get_input(0);
    const auto scale = context.get_input(1);
    const auto zero_point = context.get_input(2);
    const auto dtype = context.get_input(3);
    return {context.mark_node(quantize_per_tensor(context.get_decoder(), input, scale, zero_point, dtype))};
}

OutputVector translate_dequantize(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto input = context.get_input(0);
    return {context.mark_node(dequantize(context.get_decoder(), input))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
