// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/hammingwindow.hpp"

#include <memory>

#include "default_opset.hpp"
#include "utils/common.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector hammingwindow(const Node& node) {
    const auto size = node.get_ng_inputs().at(0);
    const auto output_datatype =
        common::get_ngraph_element_type(node.get_attribute_value<int64_t>("output_datatype", 1));
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1) == 1;

    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    // Weights as described in ONNX HammingWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#hammingwindow
    const auto float_size = std::make_shared<default_opset::Convert>(size, ov::element::f32);
    const auto a_0 = std::make_shared<default_opset::Divide>(
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{25.0f}),
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{46.0f}));
    const auto a_1 = std::make_shared<default_opset::Subtract>(
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{1.0f}),
        a_0);

    const auto start =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.0f});
    const auto one_const =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{1.0f});
    const auto two_const =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{2.0f});
    const auto range = std::make_shared<default_opset::Range>(start, size, one_const, ov::element::f32);
    const auto pi =
        default_opset::Constant::create(ov::element::f32, ov::Shape(), std::vector<float>{static_cast<float>(M_PI)});
    std::shared_ptr<ov::Node> factor;
    if (periodic) {
        factor = std::make_shared<default_opset::Multiply>(
            range,
            std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, two_const),
                                                    float_size));
    } else {
        factor = std::make_shared<default_opset::Multiply>(
            range,
            std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, two_const),
                                                    std::make_shared<default_opset::Subtract>(float_size, one_const)));
    }

    const auto cos = std::make_shared<default_opset::Cos>(factor);
    const auto scaled_cos = std::make_shared<default_opset::Multiply>(cos, a_1);
    const auto y_values = std::make_shared<default_opset::Subtract>(a_0, scaled_cos);
    if (output_datatype == element::f32) {
        return {y_values};
    } else {
        return {std::make_shared<default_opset::Convert>(y_values, output_datatype)};
    }
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END