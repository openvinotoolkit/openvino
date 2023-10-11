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
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1);

    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    // Weights as described in ONNX HammingWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#hammingwindow
    const auto a_0 = std::make_shared<default_opset::Divide>(
        std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{25.0f}),
        std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{46.0f}));
    const auto a_1 = std::make_shared<default_opset::Subtract>(
        std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{1.0f}),
        a_0);

    const auto start =
        std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{0.0f});
    const auto step = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{1.0f});
    const auto range = std::make_shared<default_opset::Range>(start, size, step, output_datatype);
    const auto pi = default_opset::Constant::create(output_datatype, ov::Shape(), {static_cast<float>(M_PI)});
    const auto size_cast = std::make_shared<default_opset::Convert>(size, output_datatype);
    const auto factor = std::make_shared<default_opset::Multiply>(
        range,
        std::make_shared<default_opset::Divide>(
            std::make_shared<default_opset::Multiply>(
                pi,
                std::make_shared<default_opset::Convert>(
                    std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{2}),
                    output_datatype)),
            periodic
                ? size_cast
                : std::make_shared<default_opset::Subtract>(
                      size_cast,
                      std::make_shared<default_opset::Convert>(
                          std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{1}),
                          output_datatype))));

    const auto cos = std::make_shared<default_opset::Cos>(factor);
    const auto scaled_cos = std::make_shared<default_opset::Multiply>(cos, a_1);
    const auto y_values = std::make_shared<default_opset::Subtract>(a_0, scaled_cos);

    return {y_values};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END