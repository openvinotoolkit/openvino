// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/shrink.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector shrink(const Node& node) {
    const auto input = node.get_ng_inputs().at(0);
    const float bias = node.get_attribute_value<float>("bias", 0.0f);
    const float lambd = node.get_attribute_value<float>("lambd", 0.5f);

    CHECK_VALID_NODE(node, !(lambd < 0.0f), " The provided 'lambd' value: ", lambd, " must not be negative.");

    std::shared_ptr<default_opset::Constant> negative_lambd;
    const auto input_element_type = input.get_element_type();
    if (input_element_type.is_signed()) {
        negative_lambd = default_opset::Constant::create(input_element_type, Shape{}, {-lambd});
    } else {
        // Passing -lambd to unsigned type constant will cause an overflow.
        // For unsigned types the lowest possible value is 0.
        negative_lambd = default_opset::Constant::create(input_element_type, Shape{}, {0});
    }

    const auto positive_lambd = default_opset::Constant::create(input_element_type, Shape{}, {lambd});

    const auto bias_tensor = default_opset::Constant::create(input_element_type, Shape{}, {bias});

    // Create a mask indicating locations of values that need to be adjusted
    // by adding and subtracting bias
    // All other values indicated by 'false' in the masks need to be zeroed out
    std::shared_ptr<ngraph::Node> values_below_neg_lambd = std::make_shared<default_opset::Less>(input, negative_lambd);
    std::shared_ptr<ngraph::Node> values_above_pos_lambd =
        std::make_shared<default_opset::Greater>(input, positive_lambd);

    // Convert from bool to the input type to be able to multiply adjusted inputs
    // by the created masks
    values_below_neg_lambd = std::make_shared<default_opset::Convert>(values_below_neg_lambd, input_element_type);
    values_above_pos_lambd = std::make_shared<default_opset::Convert>(values_above_pos_lambd, input_element_type);

    std::shared_ptr<ngraph::Node> input_minus_bias = std::make_shared<default_opset::Subtract>(input, bias_tensor);
    std::shared_ptr<ngraph::Node> input_plus_bias = std::make_shared<default_opset::Add>(input, bias_tensor);

    // multiply by the corresponding mask to zero-out the values within
    // the <-lambd;lambd> range and keep the bias-adjusted values from outside of it
    input_minus_bias = std::make_shared<default_opset::Multiply>(values_above_pos_lambd, input_minus_bias);

    input_plus_bias = std::make_shared<default_opset::Multiply>(values_below_neg_lambd, input_plus_bias);

    return {std::make_shared<default_opset::Add>(input_plus_bias, input_minus_bias)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
