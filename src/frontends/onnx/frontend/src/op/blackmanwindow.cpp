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

    element::Type tensor_type;
    switch (output_datatype) {
    case element::Type_t::f32:
        tensor_type = element::f32;
        break;
    case element::Type_t::u8:
        tensor_type = element::u8;
        break;
    case element::Type_t::i8:
        tensor_type = element::i8;
        break;
    case element::Type_t::u16:
        tensor_type = element::u16;
        break;
    case element::Type_t::i16:
        tensor_type = element::i16;
        break;
    case element::Type_t::i32:
        tensor_type = element::i32;
        break;
    case element::Type_t::i64:
        tensor_type = element::i64;
        break;
    case element::Type_t::f16:
        tensor_type = element::f16;
        break;
    case element::Type_t::f64:
        tensor_type = element::f64;
        break;
    case element::Type_t::u32:
        tensor_type = element::u32;
        break;
    case element::Type_t::u64:
        tensor_type = element::u64;
        break;
    case element::Type_t::bf16:
        tensor_type = element::bf16;
        break;
    default:
        throw std::runtime_error("Unsupported output data type.");
    }

    // Weights as described in ONNX BlackManWindow docs
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#blackmanwindow
    const auto a_0 = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<float>{0.42});
    const auto a_1 = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<float>{-0.50});
    const auto a_2 = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<float>{0.08});

    const auto start = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<float>{0.0});
    const auto step = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<float>{1.0});
    const auto range = std::make_shared<default_opset::Range>(start, size, step, tensor_type);
    const auto pi = default_opset::Constant::create(tensor_type, ov::Shape(), {static_cast<float>(M_PI)});
    const auto factor_1 = std::make_shared<default_opset::Multiply>(
        range,
        std::make_shared<default_opset::Divide>(
            std::make_shared<default_opset::Multiply>(
                pi,
                std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<int>{2})),
            periodic ? size
                     : std::make_shared<default_opset::Subtract>(
                           size,
                           std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<int>{1}))));
    const auto factor_2 = std::make_shared<default_opset::Multiply>(
        range,
        std::make_shared<default_opset::Divide>(
            std::make_shared<default_opset::Multiply>(
                pi,
                std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<int>{4})),
            periodic ? size
                     : std::make_shared<default_opset::Subtract>(
                           size,
                           std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(), std::vector<int>{1}))));
    const auto cos_1 = std::make_shared<default_opset::Cos>(factor_1);
    const auto cos_2 = std::make_shared<default_opset::Cos>(factor_2);
    const auto scaled_cos_1 = std::make_shared<default_opset::Multiply>(cos_1, a_1);
    const auto scaled_cos_2 = std::make_shared<default_opset::Multiply>(cos_2, a_2);
    const auto y_values =
        std::make_shared<default_opset::Add>(std::make_shared<default_opset::Add>(a_0, scaled_cos_1), scaled_cos_2);
    const auto output = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(axis_lengths), y_values);

    return {output};
}
}  // namespace set_17
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph