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
    const auto dtype = context.mark_node(v0::Constant::create(element::i64, Shape{}, {12}));  // qint8

    const auto three = context.mark_node(v0::Constant::create(element::f32, Shape{}, {3}));
    const auto neg_three = context.mark_node(v0::Constant::create(element::f32, Shape{}, {-3}));
    const auto six = context.mark_node(v0::Constant::create(element::f32, Shape{}, {6}));

    const auto x_dequantize = dequantize(context.get_decoder(), x);
    const auto quantized_hardswish = context.mark_node(std::make_shared<v4::HSwish>(x_dequantize));

    return {
        context.mark_node(quantize_per_tensor(context.get_decoder(), quantized_hardswish, scale, zero_point, dtype))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
