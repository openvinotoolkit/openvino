// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

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
    // const auto dtype = context.get_input(3); Not supported

    const auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));

    const auto all_axes = get_axes_range(context, 0);
    const auto max = context.mark_node(std::make_shared<opset10::ReduceMax>(input, all_axes));
    const auto min = context.mark_node(std::make_shared<opset10::ReduceMean>(input, all_axes));

    const auto zero_point_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(zero_point, max));
    const auto output_max = context.mark_node(std::make_shared<opset10::Add>(max, zero_point_convert));
    const auto output_min = context.mark_node(std::make_shared<opset10::Add>(min, zero_point_convert));

    const auto diff = context.mark_node(std::make_shared<opset10::Subtract>(max, min));
    const auto scale_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(scale, diff));
    const auto div = context.mark_node(std::make_shared<opset10::Divide>(diff, scale_convert));
    const auto one_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(one, div));
    const auto levels = context.mark_node(std::make_shared<opset10::Add>(div, one_convert));

    // const auto dtype_size = context.mark_node(v0::Constant::create(element::i32, Shape{}, {32}));
    // const auto dtype_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(dtype_size, levels));
    // const auto step_size = context.mark_node(std::make_shared<opset10::Divide>(dtype_convert, levels));

    // Affine quantization
    const auto scaled_input = context.mark_node(std::make_shared<opset10::Divide>(input, scale_convert));
    const auto scaled_input_with_zero_pt =
        context.mark_node(std::make_shared<opset10::Add>(scaled_input, zero_point_convert));
    const auto quantized_input = context.mark_node(
        std::make_shared<opset10::Round>(scaled_input_with_zero_pt, opset10::Round::RoundMode::HALF_TO_EVEN));

    const auto convert = context.mark_node(std::make_shared<opset10::Convert>(quantized_input, element::i32));
    return {convert};
}

OutputVector translate_dequantize(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto input = context.get_input(0);
    const auto convert = context.mark_node(std::make_shared<opset10::Convert>(input, element::f32));
    return {convert};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov