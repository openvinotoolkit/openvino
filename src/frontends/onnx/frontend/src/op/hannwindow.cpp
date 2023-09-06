// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/hannwindow.hpp"

#include <memory>

#include "default_opset.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_17 {
OutputVector hannwindow(const Node& node) {
    const auto size = node.get_ng_inputs().at(0);
    const auto output_datatype = common::get_ngraph_element_type(node.get_attribute_value("output_datatype", 1));
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1);

    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    // Weights as described in ONNX BlackManWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#hannwindow
    const auto a_0 = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{0.5});
    const auto a_1 = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{0.5});

    const auto start = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{0.0});
    const auto step = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{1.0});
    const auto range = std::make_shared<default_opset::Range>(start, size, step, output_datatype);
    const auto pi = default_opset::Constant::create(output_datatype, ov::Shape(), {static_cast<float>(M_PI)});
    const auto factor = std::make_shared<default_opset::Multiply>(
        range,
        std::make_shared<default_opset::Divide>(
            std::make_shared<default_opset::Multiply>(
                pi,
                std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{2})),
            periodic ? size
                     : std::make_shared<default_opset::Subtract>(
                           size,
                           std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{1}))));

    const auto cos = std::make_shared<default_opset::Cos>(factor);
    const auto scaled_cos = std::make_shared<default_opset::Multiply>(cos, a_1);
    const auto y_values = std::make_shared<default_opset::Subtract>(a_0, scaled_cos);

    return {y_values};
}
}  // namespace set_17
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph