// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/dynamic_quantize_linear.hpp"

#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"
#include "onnx_import/core/null_node.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace {
std::shared_ptr<ngraph::Node> find_min_value(const ov::Output<ov::Node>& input) {
    const auto& zero_node = default_opset::Constant::create(element::i64, Shape{}, {0});
    const auto& one_node = default_opset::Constant::create(element::i64, Shape{}, {1});

    const auto& input_shape = std::make_shared<default_opset::ShapeOf>(input);
    const auto& input_rank = std::make_shared<default_opset::ShapeOf>(input_shape);
    const auto& input_rank_as_scalar = std::make_shared<default_opset::Squeeze>(input_rank);

    const auto& reduce_axes =
        std::make_shared<default_opset::Range>(zero_node, input_rank_as_scalar, one_node, element::i64);

    const auto& input_min = std::make_shared<default_opset::ReduceMin>(input, reduce_axes);

    const auto& zero_node_u8 = default_opset::Constant::create(element::f32, Shape{}, {0});
    return std::make_shared<default_opset::Minimum>(zero_node_u8, input_min);
}

std::shared_ptr<ngraph::Node> find_max_value(const ov::Output<ov::Node>& input) {
    const auto& zero_node = default_opset::Constant::create(element::i64, Shape{}, {0});
    const auto& one_node = default_opset::Constant::create(element::i64, Shape{}, {1});

    const auto& input_shape = std::make_shared<default_opset::ShapeOf>(input);
    const auto& input_rank = std::make_shared<default_opset::ShapeOf>(input_shape);
    const auto& input_rank_as_scalar = std::make_shared<default_opset::Squeeze>(input_rank);

    const auto& reduce_axes =
        std::make_shared<default_opset::Range>(zero_node, input_rank_as_scalar, one_node, element::i64);

    const auto& input_max = std::make_shared<default_opset::ReduceMax>(input, reduce_axes);

    const auto& zero_node_u8 = default_opset::Constant::create(element::f32, Shape{}, {0});
    return std::make_shared<default_opset::Maximum>(zero_node_u8, input_max);
}

std::shared_ptr<ngraph::Node> quantize_linear(Output<ngraph::Node> x,
                                              Output<ngraph::Node> x_span,
                                              Output<ngraph::Node> quant_range_span,
                                              Output<ngraph::Node> y_zero_point) {
    const auto& x_scaled =
        std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(x, quant_range_span), x_span);

    const auto& x_rounded =
        std::make_shared<default_opset::Round>(x_scaled, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);

    const auto& y_zero_point_f32 = std::make_shared<default_opset::Convert>(y_zero_point, ov::element::f32);

    const auto& result_shifted = std::make_shared<default_opset::Add>(x_rounded, y_zero_point_f32);
    const auto& result_clamped = std::make_shared<default_opset::Clamp>(result_shifted, 0, 255);

    return std::make_shared<default_opset::Convert>(result_clamped, ov::element::u8);
}
}  // namespace
namespace op {
namespace set_1 {
OutputVector dynamic_quantize_linear(const Node& node) {
    const OutputVector& inputs = node.get_ng_inputs();
    const auto& x = inputs.at(0);

    // quantization range in case of uint8 is [0, 255]
    const auto& quant_range_min = default_opset::Constant::create(element::f32, Shape{}, {0});
    const auto& quant_range_max = default_opset::Constant::create(element::f32, Shape{}, {255});
    const auto& quant_range_span = std::make_shared<default_opset::Subtract>(quant_range_max, quant_range_min);

    const auto& x_max = find_max_value(x);
    const auto& x_min = find_min_value(x);
    const auto& x_span = std::make_shared<default_opset::Subtract>(x_max, x_min);

    const auto& y_scale = std::make_shared<default_opset::Divide>(x_span, quant_range_max);

    const auto& x_min_shifted = std::make_shared<default_opset::Subtract>(quant_range_min, x_min);
    const auto& intermediate_zero_point =
        std::make_shared<default_opset::Round>(std::make_shared<default_opset::Divide>(x_min_shifted, y_scale),
                                               ov::op::v5::Round::RoundMode::HALF_TO_EVEN);

    const auto& y_zero_point = std::make_shared<default_opset::Convert>(
        std::make_shared<default_opset::Clamp>(intermediate_zero_point, 0, 255),
        ov::element::u8);

    const auto& y = quantize_linear(x, x_span, quant_range_span, y_zero_point);

    return {y, y_scale, y_zero_point};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
