// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hswish.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantized_hardswish(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    const auto x = context.get_input(0);
    const auto scale = context.get_input(1);
    const auto zero_point = context.get_input(2);

    const auto x_dequantize = context.mark_node(dequantize(context, x.get_node_shared_ptr()));
    const auto quantized_hardswish = context.mark_node(std::make_shared<v4::HSwish>(x_dequantize));

    return {context.mark_node(quantize(context,
                                       quantized_hardswish,
                                       scale.get_node_shared_ptr(),
                                       zero_point.get_node_shared_ptr(),
                                       x.get_node_shared_ptr()))};  // take dtype from x
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
