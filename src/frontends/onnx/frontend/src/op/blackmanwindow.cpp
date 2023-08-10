// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/blackmanwindow.hpp"

#include <memory>

#include "utils/common.hpp"
#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector blackmanwindow(const Node& node) {
    const auto size = node.get_ng_inputs().at(0);
    const auto output_datatype = node.get_attribute_value<int64_t>("output_datatype", 1);
    const bool periodic = node.get_attribute_value<int64_t>("periodic", 1);

    const int64_t a_0 = 0.42, a_1 = -0.5, a_2 = 0.08;
    const ov::PartialShape shape = size.get_partial_shape();
    const std::vector<size_t> axis_lengths = shape.to_shape();

    element::Type tensor_type;
    switch (output_datatype) {
        case 1:
            tensor_type = element::f32;
            break;
        case 2:
            tensor_type = element::u8;
            break;
        case 3:
            tensor_type = element::i8;
            break;
        case 4:
            tensor_type = element::u16;
            break;
        case 5:
            tensor_type = element::i16;
            break;
        case 6:
            tensor_type = element::i32;
            break;
        case 7:
            tensor_type = element::i64;
            break;
        case 10:
            tensor_type = element::f16;
            break;
        case 11:
            tensor_type = element::f64;
            break;
        case 12:
            tensor_type = element::u32;
            break;
        case 13:
            tensor_type = element::u64;
            break;
        case 16:
            tensor_type = element::bf16;
            break;
        default:
            throw std::runtime_error("Unsupported output data type.");
    }

    if (periodic) {
        const auto range = std::make_shared<default_opset::Range>(tensor_type, size, 0, 1);
        const auto pi = default_opset::Constant::create(tensor_type, ov::Shape(), {static_cast<float>(M_PI)});
        const auto factor_1 = std::make_shared<default_opset::Multiply>(range, std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, 2), size));
        const auto factor_2 = std::make_shared<default_opset::Multiply>(range, std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, 4), size));
        const auto cos_1 = std::make_shared<default_opset::Cos>(factor_1);
        const auto cos_2 = std::make_shared<default_opset::Cos>(factor_2);
        const auto scaled_cos_1 = std::make_shared<default_opset::Multiply>(cos_1, a_1);
        const auto scaled_cos_2 = std::make_shared<default_opset::Multiply>(cos_2, a_2);
        const auto y_values = std::make_shared<default_opset::Add>(std::make_shared<default_opset::Add>(a_0, scaled_cos_1), scaled_cos_2);
        const auto output = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(axis_lengths), y_values);
        
        return {output};
    } else {
        const auto range = std::make_shared<default_opset::Range>(tensor_type, size, 0, 1);
        const auto pi = default_opset::Constant::create(tensor_type, ov::Shape(), {static_cast<float>(M_PI)});
        const auto factor_1 = std::make_shared<default_opset::Multiply>(range, std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, 2), std::make_shared<default_opset::Subtract>(size, 1)));
        const auto factor_2 = std::make_shared<default_opset::Multiply>(range, std::make_shared<default_opset::Divide>(std::make_shared<default_opset::Multiply>(pi, 4), std::make_shared<default_opset::Subtract>(size, 1)));
        const auto cos_1 = std::make_shared<default_opset::Cos>(factor_1);
        const auto cos_2 = std::make_shared<default_opset::Cos>(factor_2);
        const auto scaled_cos_1 = std::make_shared<default_opset::Multiply>(cos_1, a_1);
        const auto scaled_cos_2 = std::make_shared<default_opset::Multiply>(cos_2, a_2);
        const auto y_values = std::make_shared<default_opset::Add>(std::make_shared<default_opset::Add>(a_0, scaled_cos_1), scaled_cos_2);
        const auto output = std::make_shared<default_opset::Constant>(tensor_type, ov::Shape(axis_lengths), y_values);

        return {output};
    }
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph