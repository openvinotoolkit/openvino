// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace detail {
namespace {
ov::Output<ov::Node> get_zero_point(const ov::OutputVector& inputs) {
    if (inputs.size() > 2) {
        return inputs.at(2);
    } else {
        return std::make_shared<v0::Constant>(ov::element::u8, ov::Shape{1}, std::uint8_t(0));
    }
}

void validate_zero_point_type(const Node& onnx_node, const ov::Output<ov::Node>& y_zero_point) {
    const auto& y_zero_point_et = y_zero_point.get_element_type();
    CHECK_VALID_NODE(
        onnx_node,
        y_zero_point_et.is_static() && (y_zero_point_et == ov::element::u8 || y_zero_point_et == ov::element::i8 ||
                                        y_zero_point_et == ov::element::u16 || y_zero_point_et == ov::element::i16),
        "\"y_zero_point\" input data for QuantizeLinear operator must be one of the supported types: u8, i8, u16 or i16"
        "integer type.");
}

ov::Output<ov::Node> validate_scale(const Node& onnx_node, const ov::Output<ov::Node>& y_scale) {
    const auto& y_scale_et = y_scale.get_element_type();
    CHECK_VALID_NODE(onnx_node, y_scale_et.is_static(), "\"y_scale\" input data type must be static.");
    if (y_scale_et != ov::element::f32) {
        return std::make_shared<v0::Convert>(y_scale, ov::element::f32);
    }
    return y_scale;
}

ov::Output<ov::Node> validate_data(const Node& onnx_node, const ov::Output<ov::Node>& data) {
    const auto& data_et = data.get_element_type();
    CHECK_VALID_NODE(onnx_node, data_et.is_static(), "\"x\" input data type must be static.");

    if (data_et != ov::element::f32) {
        return std::make_shared<v0::Convert>(data, ov::element::f32);
    }
    return data;
}

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> get_output_bands(
    const ov::element::Type& destination_type,
    const ov::element::Type& data_type) {
    std::shared_ptr<ov::Node> output_low;
    std::shared_ptr<ov::Node> output_high;

    // These values could be used in a ConvertQuantizeDequantize transformation and
    // should be aligned
    switch (destination_type) {
    case ov::element::i8:
        output_low = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, -128);
        output_high = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, 127);
        break;
    case ov::element::u8:
        output_low = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, 0);
        output_high = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, 255);
        break;
    case ov::element::i16:
        output_low = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, -32768);
        output_high = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, 32767);
        break;
    case ov::element::u16:
        output_low = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, 0);
        output_high = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, 65535);
        break;
    default:
        OPENVINO_THROW("Unsupported element type for QuantizeLinear");
        break;
    }

    return std::make_tuple(output_low, output_high);
}

std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> get_input_bands(
    const ov::Output<ov::Node>& y_scale,
    const ov::Output<ov::Node>& y_zero_point,
    const std::shared_ptr<ov::Node>& output_low,
    const std::shared_ptr<ov::Node>& output_high,
    const ov::element::Type& data_type) {
    std::shared_ptr<ov::Node> input_low;
    std::shared_ptr<ov::Node> input_high;
    const auto& zero_point = std::make_shared<v0::Convert>(y_zero_point, data_type);

    input_low = std::make_shared<v1::Multiply>(y_scale, std::make_shared<v1::Subtract>(output_low, zero_point));
    if (auto constant = ov::util::get_constant_from_source(input_low)) {
        input_low = constant;
    }
    input_high = std::make_shared<v1::Multiply>(y_scale, std::make_shared<v1::Subtract>(output_high, zero_point));
    if (auto constant = ov::util::get_constant_from_source(input_high)) {
        input_high = constant;
    }

    return std::make_tuple(input_low, input_high);
}
}  // namespace
std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& y_scale,
                                             const ov::Output<ov::Node>& y_zero_point,
                                             const ov::Output<ov::Node>& data) {
    const ov::element::Type& destination_type = y_zero_point.get_element_type();
    const ov::element::Type& data_type = data.get_element_type();

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

namespace opset_1 {
ov::OutputVector quantize_linear(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};
    auto x = inputs.at(0);
    auto y_scale = inputs.at(1);
    auto y_zero_point = detail::get_zero_point(inputs);

    x = detail::validate_data(node, x);
    detail::validate_zero_point_type(node, y_zero_point);
    y_scale = detail::validate_scale(node, y_scale);

    return {detail::make_fake_quantize(y_scale, y_zero_point, x)};
}
ONNX_OP("QuantizeLinear", {1, 12}, ai_onnx::opset_1::quantize_linear);
}  // namespace opset_1

namespace opset_13 {
namespace detail {
ov::OutputVector quantize_linear(ov::Output<ov::Node> x,
                                 ov::Output<ov::Node> y_scale,
                                 ov::Output<ov::Node> y_zero_point,
                                 int64_t axis,
                                 Node node) {
    namespace detail = ov::frontend::onnx::ai_onnx::detail;

    x = detail::validate_data(node, x);
    detail::validate_zero_point_type(node, y_zero_point);
    y_scale = detail::validate_scale(node, y_scale);

    const auto& x_shape = x.get_partial_shape();

    if (x_shape.rank().is_static()) {
        axis = common::normalize_axis(node.get_description(), axis, x_shape.rank());
    }

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

        ov::Shape target_shape(x_shape.rank().get_length(), 1);
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

        ov::Shape target_shape(x_shape.rank().get_length(), 1);
        target_shape[axis] = static_cast<size_t>(x_shape[axis].get_length());

        y_zero_point = ov::op::util::reshape(y_zero_point, target_shape);
    }

    return {detail::make_fake_quantize(y_scale, y_zero_point, x)};
}
}  // namespace detail

ov::OutputVector quantize_linear(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector inputs{node.get_ov_inputs()};

    FRONT_END_GENERAL_CHECK(2 <= inputs.size() && inputs.size() <= 3,
                            "The QuantizeLinear op expects 2 required and one optional "
                            "input. Got: ",
                            inputs.size());

    const auto& x = inputs[0];
    const auto& scale = inputs[1];
    const auto zero_point = ai_onnx::detail::get_zero_point(inputs);

    // per-tensor quantization, axis attribute ignored
    if (scale.get_partial_shape().rank().is_static() && scale.get_partial_shape().rank().get_length() == 0 &&
        zero_point.get_partial_shape().rank().is_static() && zero_point.get_partial_shape().rank().get_length() == 0) {
        return ai_onnx::opset_1::quantize_linear(node);
    }

    return detail::quantize_linear(x, scale, zero_point, node.get_attribute_value<int64_t>("axis", 1), node);
}
ONNX_OP("QuantizeLinear", OPSET_SINCE(13), ai_onnx::opset_13::quantize_linear);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
