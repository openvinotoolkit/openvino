// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantized_mul(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto x = context.get_input(0);
    const auto y = context.get_input(1);
    const auto scale = context.get_input(2);
    const auto zero_point = context.get_input(3);

    const auto quantized_mul = context.mark_node(std::make_shared<v1::Multiply>(x, y));

    return {quantize(context, quantized_mul, scale, zero_point, x)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
