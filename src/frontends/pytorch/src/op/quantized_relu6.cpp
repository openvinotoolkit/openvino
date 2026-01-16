// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/clamp.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_quantized_relu6(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    const auto x = context.get_input(0);

    const auto quantized_relu6 = context.mark_node(std::make_shared<ov::op::v0::Clamp>(x, 0., 6.));

    return {quantize(context, quantized_relu6, x)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
