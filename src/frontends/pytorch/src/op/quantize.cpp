// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/subtract.hpp"
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

    const auto scale_convert = context.mark_node(std::make_shared<v1::ConvertLike>(scale, input));
    const auto zero_point_convert = context.mark_node(std::make_shared<v1::ConvertLike>(zero_point, input));
    const auto scaled_input = context.mark_node(std::make_shared<v1::Divide>(input, scale_convert));
    const auto scaled_input_with_zero_pt =
        context.mark_node(std::make_shared<v1::Add>(scaled_input, zero_point_convert));
    const auto quantized_input =
        context.mark_node(std::make_shared<v5::Round>(scaled_input_with_zero_pt, v5::Round::RoundMode::HALF_TO_EVEN));

    ov::Output<ov::Node> output;
    if (dtype == element::u8) {
        const auto clamp = context.mark_node(std::make_shared<v0::Clamp>(quantized_input,
                                                                         std::numeric_limits<unsigned char>::lowest(),
                                                                         std::numeric_limits<unsigned char>::max()));
        output = context.mark_node(std::make_shared<v0::Convert>(clamp, element::u8));
    } else if (dtype == element::i8) {
        const auto clamp = context.mark_node(std::make_shared<v0::Clamp>(quantized_input,
                                                                         std::numeric_limits<char>::lowest(),
                                                                         std::numeric_limits<char>::max()));
        output = context.mark_node(std::make_shared<v0::Convert>(clamp, element::i8));
    } else {
        output = context.mark_node(std::make_shared<v0::Convert>(quantized_input, element::i32));
    }

    const auto quantized_pt_node =
        context.mark_node(std::make_shared<QuantizedPtNode>(QuantizedPtNodeType::QUANTIZE_PER_TENSOR,
                                                            context,
                                                            output,
                                                            scale,
                                                            zero_point));
    return {quantized_pt_node};
}

OutputVector translate_quantize_per_channel(const NodeContext& context) {
    num_inputs_check(context, 5, 5);
    const auto input = context.get_input(0);
    const auto scales = context.get_input(1);
    const auto zero_points = context.get_input(2);
    const auto axis = context.get_input(3);
    const auto dtype = convert_dtype(context.const_input<int64_t>(4));
    // TODO
    const auto quantized_pt_node =
        context.mark_node(std::make_shared<QuantizedPtNode>(QuantizedPtNodeType::QUANTIZE_PER_CHANNEL,
                                                            context,
                                                            input,
                                                            scales,
                                                            zero_points,
                                                            axis));
    return {quantized_pt_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
