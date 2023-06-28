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
    const auto dtype = convert_dtype(context.const_input<int64_t>(3));

    // Affine quantization
    const auto scale_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(scale, input));
    const auto zero_point_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(zero_point, input));
    const auto scaled_input = context.mark_node(std::make_shared<opset10::Divide>(input, scale_convert));
    const auto scaled_input_with_zero_pt =
        context.mark_node(std::make_shared<opset10::Add>(scaled_input, zero_point_convert));
    const auto quantized_input = context.mark_node(
        std::make_shared<opset10::Round>(scaled_input_with_zero_pt, opset10::Round::RoundMode::HALF_TO_EVEN));

    std::shared_ptr<ov::Node> output;
    if (dtype == element::u8) {
        const auto clamp =
            context.mark_node(std::make_shared<opset10::Clamp>(quantized_input,
                                                               std::numeric_limits<unsigned char>::lowest(),
                                                               std::numeric_limits<unsigned char>::max()));
        output = context.mark_node(std::make_shared<opset10::Convert>(clamp, element::u8));
    } else if (dtype == element::i8) {
        const auto clamp = context.mark_node(std::make_shared<opset10::Clamp>(quantized_input,
                                                                              std::numeric_limits<char>::lowest(),
                                                                              std::numeric_limits<char>::max()));
        output = context.mark_node(std::make_shared<opset10::Convert>(clamp, element::i8));
    } else {
        output = context.mark_node(std::make_shared<opset10::Convert>(quantized_input, element::i32));
    }
    return {output};
}

OutputVector translate_dequantize(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto input = context.get_input(0);
    const auto convert = context.mark_node(std::make_shared<opset10::Convert>(input, element::f32));
    // const auto zero_point = ?
    // const auto scale = ?

    // const auto scale_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(scale, input));
    // const auto zero_point_convert = context.mark_node(std::make_shared<opset10::ConvertLike>(zero_point, input));

    // const auto input_without_zero_pt =
    //     context.mark_node(std::make_shared<opset10::Subtract>(convert, zero_point_convert));
    // const auto dequantized_input = context.mark_node(std::make_shared<opset10::Multiply>(input_without_zero_pt,
    // scale_convert));

    return {convert};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
