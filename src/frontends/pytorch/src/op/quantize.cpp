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
    const auto dtype = convert_dtype(context.const_input<int64_t>(3));
    return {quantize(context, input, scale, zero_point, dtype, QuantizedPtNodeType::QUANTIZE_PER_TENSOR)};
}

OutputVector translate_quantize_per_channel(const NodeContext& context) {
    num_inputs_check(context, 5, 5);
    const auto input = context.get_input(0);
    const auto scales = context.get_input(1);
    const auto zero_points = context.get_input(2);
    const auto axis = context.get_input(3);
    const auto dtype = convert_dtype(context.const_input<int64_t>(4));
    return {quantize(context, input, scales, zero_points, axis, dtype, QuantizedPtNodeType::QUANTIZE_PER_CHANNEL)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
