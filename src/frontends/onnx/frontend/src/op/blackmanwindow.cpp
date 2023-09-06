// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/blackmanwindow.hpp"

#include <memory>

#include "default_opset.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_17 {
OutputVector blackmanwindow(const Node& node) {
    const auto size = node.get_ng_inputs().at(0);
    const auto output_datatype = common::get_ngraph_element_type(node.get_attribute_value("output_datatype", 1));
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1);

    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    // Weights as described in ONNX BlackManWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#blackmanwindow
    const auto a_0 = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{0.42});
    const auto a_1 = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{-0.50});
    const auto a_2 = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{0.08});

    const auto start = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{0.0});
    const auto step = std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<float>{1.0});
    const auto range = std::make_shared<default_opset::Range>(start, size, step, output_datatype);
    const auto pi = default_opset::Constant::create(output_datatype, ov::Shape(), {static_cast<float>(M_PI)});
    const auto factor_1 = std::make_shared<default_opset::Multiply>(
        range,
        std::make_shared<default_opset::Divide>(
            std::make_shared<default_opset::Multiply>(
                pi,
                std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{2})),
            periodic
                ? size
                : std::make_shared<default_opset::Subtract>(
                      size,
                      std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{1}))));
    const auto factor_2 = std::make_shared<default_opset::Multiply>(
        range,
        std::make_shared<default_opset::Divide>(
            std::make_shared<default_opset::Multiply>(
                pi,
                std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{4})),
            periodic
                ? size
                : std::make_shared<default_opset::Subtract>(
                      size,
                      std::make_shared<default_opset::Constant>(output_datatype, ov::Shape(), std::vector<int>{1}))));
    const auto cos_1 = std::make_shared<default_opset::Cos>(factor_1);
    const auto cos_2 = std::make_shared<default_opset::Cos>(factor_2);
    const auto scaled_cos_1 = std::make_shared<default_opset::Multiply>(cos_1, a_1);
    const auto scaled_cos_2 = std::make_shared<default_opset::Multiply>(cos_2, a_2);
    const auto y_values =
        std::make_shared<default_opset::Add>(std::make_shared<default_opset::Add>(a_0, scaled_cos_1), scaled_cos_2);

    return {y_values};
}
}  // namespace set_17
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph