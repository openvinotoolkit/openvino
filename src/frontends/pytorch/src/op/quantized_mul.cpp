// Copyright (C) 2018-2023 Intel Corporation
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
    const auto dtype =
        x.get_node_shared_ptr()->get_input_node_ptr(x.get_node_shared_ptr()->get_input_size() - 1)->output(0);
    // const auto dtype = context.mark_node(v0::Constant::create(element::i64, Shape{}, {12}));  // qint8

    const auto x_dequantize = context.mark_node(dequantize(context.get_decoder(), x));
    const auto y_dequantize = context.mark_node(dequantize(context.get_decoder(), y));
    const auto quantized_mul = context.mark_node(std::make_shared<v1::Multiply>(x_dequantize, y_dequantize));

    return {context.mark_node(quantize_per_tensor(context.get_decoder(), quantized_mul, scale, zero_point, dtype))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
