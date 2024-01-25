// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/celu.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector celu(const Node& node) {
    double alpha = node.get_attribute_value<float>("alpha", 1.0);
    auto alpha_node = default_opset::Constant::create(element::f32, Shape{}, {alpha});
    auto zero_node = default_opset::Constant::create(element::f32, Shape{}, {0});
    auto one_node = default_opset::Constant::create(element::f32, Shape{}, {1});
    auto input = node.get_ng_inputs().at(0);
    CHECK_VALID_NODE(node, input.get_element_type() == element::f32, "Only float32 input type is supported")
    input = std::make_shared<default_opset::Convert>(input, element::f32);

    auto positive_input = std::make_shared<default_opset::Maximum>(zero_node, input);
    auto negative_input = std::make_shared<default_opset::Minimum>(
        zero_node,
        std::make_shared<default_opset::Multiply>(
            alpha_node,
            std::make_shared<default_opset::Subtract>(
                std::make_shared<default_opset::Exp>(std::make_shared<default_opset::Divide>(input, alpha_node)),
                one_node)));

    auto output = std::make_shared<default_opset::Add>(positive_input, negative_input);
    return {output};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
