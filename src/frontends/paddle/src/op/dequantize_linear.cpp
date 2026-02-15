// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs dequantize_linear(const NodeContext& node) {
    // extract the INPUTS
    const auto x = node.get_input("X");
    const auto scale = node.get_input("Scale");
    const auto zero_point = node.get_input("ZeroPoint");

    // assert shape of scale and zero_point
    const auto& scale_shape = scale.get_partial_shape();
    PADDLE_OP_CHECK(node, scale.get_partial_shape().rank().is_static(), "dequantize_linear scale rank must be static.");
    const auto& scale_shape_length = scale.get_partial_shape().rank().get_length();

    if (scale_shape_length == 1) {
        PADDLE_OP_CHECK(node,
                        scale.get_partial_shape() == zero_point.get_partial_shape(),
                        "dequantize_linear shape of scale and zero_point doesn't match.");
    } else if (scale_shape_length == 2) {
        PADDLE_OP_CHECK(node,
                        scale.get_partial_shape()[1] == zero_point.get_partial_shape()[0],
                        "dequantize_linear shape of scale and zero_point doesn't match.");
    } else {
        PADDLE_OP_CHECK(node, false, "dims of scale should not be greater than 2.");
    }

    const auto bit_length = node.get_attribute<int32_t>("bit_length");
    const auto range = (1 << (bit_length - 1)) - 1;
    const auto range_node = std::make_shared<default_opset::Constant>(element::f32, Shape{1}, (1.0 / range));
    const auto real_scale = std::make_shared<default_opset::Multiply>(scale, range_node);

    auto q_node = std::make_shared<default_opset::Convert>(x, element::f32);
    // extract the ATTRIBUTES and explaination for quant_axis:
    //             / [-1]      --- per-tensor, scale is always 1-D
    // quant_axis  - [0 or 1]  --- per-channel, scale may be 1-D or 2-D, needing to reshape for input shape.
    //             \ [others]  --- unsupported
    auto quant_axis = node.get_attribute<int32_t>("quant_axis");
    std::vector<int32_t> quant_axis_range{-1, 0, 1};
    PADDLE_OP_CHECK(node,
                    std::any_of(quant_axis_range.begin(),
                                quant_axis_range.end(),
                                [&quant_axis](int32_t value) {
                                    return quant_axis == value;
                                }),
                    "dequantize_linear quant_axis is NOT in the range of [-1, 0, 1].");
    if (quant_axis == -1) {
        const auto zp_node = std::make_shared<default_opset::Convert>(zero_point, element::f32);
        const auto out_node =
            std::make_shared<default_opset::Multiply>(std::make_shared<default_opset::Subtract>(q_node, zp_node),
                                                      real_scale);
        return node.default_single_output_mapping({out_node}, {"Y"});
    } else {
        // But for per-channel scenario, the shape of scale is NOT stable.
        // Sometimes scale is 1-D and sometimes scale is 2-D. But the last dim(e.g. s[len-1]) really makes sense.
        // Let's prepare a pattern to reshape operation according to the scale shape.
        std::vector<size_t> reshape_pattern(x.get_partial_shape().rank().get_length(), 1);
        reshape_pattern.at(quant_axis) = scale_shape[scale_shape_length - 1].get_length();
        const auto reshape_node =
            std::make_shared<default_opset::Constant>(element::i32, Shape{reshape_pattern.size()}, reshape_pattern);
        const auto reshape_scale = std::make_shared<default_opset::Reshape>(real_scale, reshape_node, true);
        const auto zp_node = std::make_shared<default_opset::Convert>(
            std::make_shared<default_opset::Reshape>(zero_point, reshape_node, true),
            element::f32);
        const auto out_node =
            std::make_shared<default_opset::Multiply>(std::make_shared<default_opset::Subtract>(q_node, zp_node),
                                                      reshape_scale);
        return node.default_single_output_mapping({out_node}, {"Y"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
