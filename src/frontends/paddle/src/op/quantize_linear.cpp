// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"

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
namespace {
ov::Output<ov::Node> get_zero_point(const NodeContext& node) {
    if (node.has_input("ZeroPoint")) {
        return node.get_input("ZeroPoint");
    } else {
        return std::make_shared<default_opset::Constant>(ov::element::i32, ov::Shape{1}, 0);
    }
}

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> get_output_bands(
    const ov::element::Type& destination_type,
    const ov::element::Type& data_type) {
    std::shared_ptr<ov::Node> output_low, output_high;

    switch (destination_type) {
    case ov::element::i8:
        output_low = std::make_shared<default_opset::Constant>(data_type, ov::Shape{1}, -128);
        output_high = std::make_shared<default_opset::Constant>(data_type, ov::Shape{1}, 127);
        break;
    case ov::element::u8:
        output_low = std::make_shared<default_opset::Constant>(data_type, ov::Shape{1}, 0);
        output_high = std::make_shared<default_opset::Constant>(data_type, ov::Shape{1}, 255);
        break;
    default:
        OPENVINO_ASSERT(false,
                        "Unsupported element type for QuantizeLinear: destination_type = ",
                        destination_type,
                        ", data_type = ",
                        data_type);
    }

    return std::make_tuple(output_low, output_high);
}

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> get_input_bands(
    const ov::Output<ov::Node>& y_scale,
    const ov::Output<ov::Node>& y_zero_point,
    const std::shared_ptr<ov::Node>& output_low,
    const std::shared_ptr<ov::Node>& output_high,
    const ov::element::Type& data_type) {
    auto zero_point = std::make_shared<default_opset::Convert>(y_zero_point, data_type);
    auto scale = std::make_shared<default_opset::Convert>(y_scale, data_type);

    auto input_low =
        std::make_shared<default_opset::Multiply>(scale,
                                                  std::make_shared<default_opset::Subtract>(output_low, zero_point));
    auto input_high =
        std::make_shared<default_opset::Multiply>(scale,
                                                  std::make_shared<default_opset::Subtract>(output_high, zero_point));

    return std::make_tuple(input_low, input_high);
}

ov::Output<ov::Node> reshape_for_broadcast(const ov::Output<ov::Node>& input, int64_t axis, const ov::Shape& x_shape) {
    if (input.get_partial_shape().rank().get_length() == 0) {
        return input;
    }

    ov::Shape target_shape(x_shape.size(), 1);
    target_shape[axis] = input.get_shape()[0];

    auto shape_const =
        std::make_shared<default_opset::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
    return std::make_shared<default_opset::Reshape>(input, shape_const, true);
}

}  // namespace

NamedOutputs quantize_linear(const NodeContext& node) {
    auto x = node.get_input("X");
    auto y_scale = node.get_input("Scale");
    auto y_zero_point = get_zero_point(node);
    auto axis = node.get_attribute<int64_t>("axis", 1);

    const auto& x_shape = x.get_partial_shape();
    PADDLE_OP_CHECK(node, x_shape.rank().is_static(), "Rank of input tensor must be static");
    axis = ov::util::normalize_axis(axis, x_shape.rank().get_length());

    const auto& destination_type = y_zero_point.get_element_type();
    const auto& data_type = x.get_element_type();

    std::shared_ptr<ov::Node> output_low, output_high;
    std::tie(output_low, output_high) = get_output_bands(destination_type, data_type);
    // Reshape scale and zero_point for per-channel quantization
    y_scale = reshape_for_broadcast(y_scale, axis, x_shape.get_shape());

    std::shared_ptr<ov::Node> input_low, input_high;
    std::tie(input_low, input_high) = get_input_bands(y_scale, y_zero_point, output_low, output_high, data_type);
    const std::size_t levels = static_cast<size_t>(1) << destination_type.bitwidth();

    auto fake_quantize =
        std::make_shared<default_opset::FakeQuantize>(x, input_low, input_high, output_low, output_high, levels);

    return node.default_single_output_mapping({fake_quantize}, {"Y"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
