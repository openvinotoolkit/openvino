// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/quantize_linear.hpp"

#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "ov_models/ov_builders/reshape.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
namespace {
Output<ov::Node> get_zero_point(const OutputVector& inputs) {
    if (inputs.size() > 2) {
        return inputs.at(2);
    } else {
        return std::make_shared<v0::Constant>(element::u8, Shape{1}, std::uint8_t(0));
    }
}

void validate_zero_point_type(const Node& onnx_node, const Output<ov::Node>& y_zero_point) {
    const auto& y_zero_point_et = y_zero_point.get_element_type();
    CHECK_VALID_NODE(
        onnx_node,
        y_zero_point_et.is_static() && (y_zero_point_et == element::u8 || y_zero_point_et == element::i8 ||
                                        y_zero_point_et == element::u16 || y_zero_point_et == element::i16),
        "\"y_zero_point\" input data for QuantizeLinear operator must be one of the supported types: u8, i8, u16 or i16"
        "integer type.");
}

Output<ov::Node> validate_scale(const Node& onnx_node, const Output<ov::Node>& y_scale) {
    const auto& y_scale_et = y_scale.get_element_type();
    CHECK_VALID_NODE(onnx_node, y_scale_et.is_static(), "\"y_scale\" input data type must be static.");
    if (y_scale_et != element::f32) {
        return std::make_shared<v0::Convert>(y_scale, element::f32);
    }
    return y_scale;
}

Output<ov::Node> validate_data(const Node& onnx_node, const Output<ov::Node>& data) {
    const auto& data_et = data.get_element_type();
    CHECK_VALID_NODE(onnx_node, data_et.is_static(), "\"x\" input data type must be static.");

    if (data_et != element::f32) {
        return std::make_shared<v0::Convert>(data, element::f32);
    }
    return data;
}

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> get_output_bands(const element::Type& destination_type,
                                                                                  const element::Type& data_type) {
    std::shared_ptr<ov::Node> output_low;
    std::shared_ptr<ov::Node> output_high;

    // These values could be used in a ConvertQuantizeDequantize transformation and
    // should be aligned
    switch (destination_type) {
    case element::i8:
        output_low = std::make_shared<v0::Constant>(data_type, Shape{1}, -128);
        output_high = std::make_shared<v0::Constant>(data_type, Shape{1}, 127);
        break;
    case element::u8:
        output_low = std::make_shared<v0::Constant>(data_type, Shape{1}, 0);
        output_high = std::make_shared<v0::Constant>(data_type, Shape{1}, 255);
        break;
    case element::i16:
        output_low = std::make_shared<v0::Constant>(data_type, Shape{1}, -32768);
        output_high = std::make_shared<v0::Constant>(data_type, Shape{1}, 32767);
        break;
    case element::u16:
        output_low = std::make_shared<v0::Constant>(data_type, Shape{1}, 0);
        output_high = std::make_shared<v0::Constant>(data_type, Shape{1}, 65535);
        break;
    default:
        OPENVINO_THROW("Unsupported element type for QuantizeLinear");
        break;
    }

    return std::make_tuple(output_low, output_high);
}

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> get_input_bands(
    const Output<ov::Node>& y_scale,
    const Output<ov::Node>& y_zero_point,
    const std::shared_ptr<ov::Node>& output_low,
    const std::shared_ptr<ov::Node>& output_high,
    const element::Type& data_type) {
    std::shared_ptr<ov::Node> input_low;
    std::shared_ptr<ov::Node> input_high;
    const auto& zero_point = std::make_shared<v0::Convert>(y_zero_point, data_type);

    input_low = std::make_shared<v1::Multiply>(y_scale, std::make_shared<v1::Subtract>(output_low, zero_point));
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (auto constant = ov::get_constant_from_source(input_low)) {
        OPENVINO_SUPPRESS_DEPRECATED_END
        input_low = constant;
    }
    input_high = std::make_shared<v1::Multiply>(y_scale, std::make_shared<v1::Subtract>(output_high, zero_point));
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (auto constant = ov::get_constant_from_source(input_high)) {
        OPENVINO_SUPPRESS_DEPRECATED_END
        input_high = constant;
    }

    return std::make_tuple(input_low, input_high);
}
}  // namespace
std::shared_ptr<ov::Node> make_fake_quantize(const Output<ov::Node>& y_scale,
                                             const Output<ov::Node>& y_zero_point,
                                             const Output<ov::Node>& data) {
    const element::Type& destination_type = y_zero_point.get_element_type();
    const element::Type& data_type = data.get_element_type();

    std::shared_ptr<ov::Node> output_low;
    std::shared_ptr<ov::Node> output_high;
    std::tie(output_low, output_high) = detail::get_output_bands(destination_type, data_type);

    std::shared_ptr<ov::Node> input_low;
    std::shared_ptr<ov::Node> input_high;
    std::tie(input_low, input_high) =
        detail::get_input_bands(y_scale, y_zero_point, output_low, output_high, data_type);

    const std::size_t levels = static_cast<size_t>(1) << destination_type.bitwidth();

    return std::make_shared<v0::Convert>(
        std::make_shared<v0::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels),
        destination_type);
}
}  // namespace detail

namespace set_1 {
OutputVector quantize_linear(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    auto x = inputs.at(0);
    auto y_scale = inputs.at(1);
    auto y_zero_point = detail::get_zero_point(inputs);

    x = detail::validate_data(node, x);
    detail::validate_zero_point_type(node, y_zero_point);
    y_scale = detail::validate_scale(node, y_scale);

    return {detail::make_fake_quantize(y_scale, y_zero_point, x)};
}
}  // namespace set_1

namespace set_13 {
namespace {
OutputVector quantize_linear(Output<ov::Node> x,
                             Output<ov::Node> y_scale,
                             Output<ov::Node> y_zero_point,
                             int64_t axis,
                             Node node) {
    namespace detail = ngraph::onnx_import::op::detail;

    x = detail::validate_data(node, x);
    detail::validate_zero_point_type(node, y_zero_point);
    y_scale = detail::validate_scale(node, y_scale);

    const auto& x_shape = x.get_partial_shape();

    OPENVINO_SUPPRESS_DEPRECATED_START
    axis = normalize_axis(node.get_description(), axis, x_shape.rank());
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto& y_scale_shape = y_scale.get_partial_shape();
    const auto& y_zero_point_shape = y_zero_point.get_partial_shape();

    if (y_scale_shape.rank().is_static() && y_scale_shape.rank().get_length() == 1 && x_shape.rank().is_static() &&
        x_shape.rank().get_length() > 0 && x_shape[axis].is_static()) {
        CHECK_VALID_NODE(node,
                         y_scale_shape[0].same_scheme(x_shape[axis]),
                         "The number of quantization scale elements ",
                         y_scale_shape[0],
                         " must match the number of respective input data axis size: ",
                         x_shape[axis]);

        Shape target_shape(x_shape.rank().get_length(), 1);
        target_shape[axis] = static_cast<size_t>(x_shape[axis].get_length());

        y_scale = ov::op::util::reshape(y_scale, target_shape);
    }

    if (y_zero_point_shape.rank().is_static() && y_zero_point_shape.rank().get_length() == 1 &&
        x_shape.rank().is_static() && x_shape.rank().get_length() > 0 && x_shape[axis].is_static()) {
        CHECK_VALID_NODE(node,
                         y_zero_point_shape[0].same_scheme(x_shape[axis]),
                         "The number of quantization zero point elements ",
                         y_zero_point_shape[0],
                         " must match the number of respective input data axis size: ",
                         x_shape[axis]);

        Shape target_shape(x_shape.rank().get_length(), 1);
        target_shape[axis] = static_cast<size_t>(x_shape[axis].get_length());

        y_zero_point = ov::op::util::reshape(y_zero_point, target_shape);
    }

    return {detail::make_fake_quantize(y_scale, y_zero_point, x)};
}
}  // namespace

OutputVector quantize_linear(const Node& node) {
    const OutputVector inputs{node.get_ng_inputs()};

    FRONT_END_GENERAL_CHECK(2 <= inputs.size() && inputs.size() <= 3,
                            "The QuantizeLinear op expects 2 required and one optional "
                            "input. Got: ",
                            inputs.size());

    const auto& x = inputs[0];
    const auto& scale = inputs[1];
    const auto zero_point = op::detail::get_zero_point(inputs);

    // per-tensor quantization, axis attribute ignored
    if (scale.get_partial_shape().rank().is_static() && scale.get_partial_shape().rank().get_length() == 0 &&
        zero_point.get_partial_shape().rank().is_static() && zero_point.get_partial_shape().rank().get_length() == 0) {
        return set_1::quantize_linear(node);
    }

    return quantize_linear(x, scale, zero_point, node.get_attribute_value<int64_t>("axis", 1), node);
}
}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
