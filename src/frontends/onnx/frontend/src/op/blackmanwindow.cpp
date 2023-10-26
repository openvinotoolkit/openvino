// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/blackmanwindow.hpp"

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
OutputVector blackmanwindow(const Node& node) {
    const auto size = node.get_ng_inputs().at(0);
    const auto output_datatype =
        common::get_ngraph_element_type(node.get_attribute_value<int64_t>("output_datatype", 1));
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1) == 1;

    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    // Weights as described in ONNX BlackmanWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#blackmanwindow
    const auto float_size = std::make_shared<default_opset::Convert>(size, ov::element::f32);
    const auto a_0 =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.42f});
    const auto a_1 =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{-0.50f});
    const auto a_2 =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.08f});

    const auto start =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{0.0f});
    const auto one_const =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{1.0f});
    const auto two_const =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{2.0f});
    const auto four_const =
        std::make_shared<default_opset::Constant>(ov::element::f32, ov::Shape(), std::vector<float>{4.0f});
    const auto range = std::make_shared<default_opset::Range>(start, size, one_const, ov::element::f32);
    const auto pi =
        default_opset::Constant::create(ov::element::f32, ov::Shape(), std::vector<float>{static_cast<float>(M_PI)});
    std::shared_ptr<ov::Node> factor_1, factor_2;
    if (periodic) {
        factor_1 = std::make_shared<default_opset::Multiply>(
            range,
            std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, two_const),
                                                    float_size));
        factor_2 = std::make_shared<default_opset::Multiply>(
            range,
            std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, four_const),
                                                    float_size));
    } else {
        factor_1 = std::make_shared<default_opset::Multiply>(
            range,
            std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, two_const),
                                                    std::make_shared<default_opset::Subtract>(float_size, one_const)));
        factor_2 = std::make_shared<default_opset::Multiply>(
            range,
            std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, four_const),
                                                    std::make_shared<default_opset::Subtract>(float_size, one_const)));
    }

    const auto cos_1 = std::make_shared<default_opset::Cos>(factor_1);
    const auto cos_2 = std::make_shared<default_opset::Cos>(factor_2);
    const auto scaled_cos_1 = std::make_shared<default_opset::Multiply>(cos_1, a_1);
    const auto scaled_cos_2 = std::make_shared<default_opset::Multiply>(cos_2, a_2);
    const auto y_values =
        std::make_shared<default_opset::Add>(std::make_shared<default_opset::Add>(a_0, scaled_cos_1), scaled_cos_2);

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