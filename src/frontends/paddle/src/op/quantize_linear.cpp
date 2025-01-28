// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
/*
    quantize_linear:
    INT ---------------------------> FLOAT
            [scale, zero_point]

    zero_point is always equal to zero.

                / [-1]      --- per-tensor
    quant_axis  - [0 or 1]  --- per-channel, expand 1-D tensor to match the input
                \ [others]  --- unsupported

                / [0]       --- rounding to nearest ties to even
    round_type  - [1]       --- rounding to nearest ties away from zero
                \ [others]  --- unsupported!
    refer to https://en.wikipedia.org/wiki/IEEE_754 for more info about round_type

*/
NamedOutputs quantize_linear(const NodeContext& node) {
    // extract the INPUTS
    const auto x = node.get_input("X");
    const auto scale = node.get_input("Scale");
    const auto zero_point = node.get_input("ZeroPoint");
    const auto quant_axis = node.get_attribute<int32_t>("quant_axis");
    const std::vector<int32_t> quant_axis_range{-1};
    PADDLE_OP_CHECK(node,
                    std::any_of(quant_axis_range.begin(),
                                quant_axis_range.end(),
                                [&quant_axis](int32_t value) {
                                    return quant_axis == value;
                                }),
                    "quantize_linear quant_axis is NOT in the range of [-1].");

    // extract the ATTRIBUTES
    const auto bit_length = node.get_attribute<int32_t>("bit_length");
    const auto range = (1 << (bit_length - 1)) - 1;
    const auto high_range = (1 << (bit_length - 1)) - 1;
    const auto low_range = -(1 << (bit_length - 1));
    const auto round_mode = [&]() {
        if (node.has_attribute("round_type")) {
            if (node.get_attribute<int32_t>("round_type")) {
                return default_opset::Round::RoundMode::HALF_AWAY_FROM_ZERO;
            } else {
                return default_opset::Round::RoundMode::HALF_TO_EVEN;
            }
        } else {
            return default_opset::Round::RoundMode::HALF_TO_EVEN;
        }
    }();

    const auto zp_node = std::make_shared<default_opset::Convert>(zero_point, element::f32);
    const auto q_sub_node = std::make_shared<default_opset::Subtract>(x, zp_node);
    const auto range_node = std::make_shared<default_opset::Constant>(element::f32, Shape{1}, (1.0 / range));
    const auto real_scale = std::make_shared<default_opset::Multiply>(scale, range_node);
    const auto q_div_node = std::make_shared<default_opset::Divide>(q_sub_node, real_scale);
    const auto q_round_node = std::make_shared<default_opset::Round>(q_div_node, round_mode);
    const auto q_node = std::make_shared<default_opset::Clamp>(q_round_node, low_range, high_range);
    return node.default_single_output_mapping({q_node}, {"Y"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
