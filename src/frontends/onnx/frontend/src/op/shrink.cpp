// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector shrink(const ov::frontend::onnx::Node& node) {
    const auto input = node.get_ov_inputs().at(0);
    const float bias = node.get_attribute_value<float>("bias", 0.0f);
    const float lambd = node.get_attribute_value<float>("lambd", 0.5f);

    CHECK_VALID_NODE(node, !(lambd < 0.0f), " The provided 'lambd' value: ", lambd, " must not be negative.");

    std::shared_ptr<v0::Constant> negative_lambd;
    const auto input_element_type = input.get_element_type();
    if (input_element_type.is_signed()) {
        negative_lambd = v0::Constant::create(input_element_type, ov::Shape{}, {-lambd});
    } else {
        // Passing -lambd to unsigned type constant will cause an overflow.
        // For unsigned types the lowest possible value is 0.
        negative_lambd = v0::Constant::create(input_element_type, ov::Shape{}, {0});
    }

    const auto positive_lambd = v0::Constant::create(input_element_type, ov::Shape{}, {lambd});

    const auto bias_tensor = v0::Constant::create(input_element_type, ov::Shape{}, {bias});

    // Create a mask indicating locations of values that need to be adjusted
    // by adding and subtracting bias
    // All other values indicated by 'false' in the masks need to be zeroed out
    std::shared_ptr<ov::Node> values_below_neg_lambd = std::make_shared<v1::Less>(input, negative_lambd);
    std::shared_ptr<ov::Node> values_above_pos_lambd = std::make_shared<v1::Greater>(input, positive_lambd);

    // Convert from bool to the input type to be able to multiply adjusted inputs
    // by the created masks
    values_below_neg_lambd = std::make_shared<v0::Convert>(values_below_neg_lambd, input_element_type);
    values_above_pos_lambd = std::make_shared<v0::Convert>(values_above_pos_lambd, input_element_type);

    std::shared_ptr<ov::Node> input_minus_bias = std::make_shared<v1::Subtract>(input, bias_tensor);
    std::shared_ptr<ov::Node> input_plus_bias = std::make_shared<v1::Add>(input, bias_tensor);

    // multiply by the corresponding mask to zero-out the values within
    // the <-lambd;lambd> range and keep the bias-adjusted values from outside of it
    input_minus_bias = std::make_shared<v1::Multiply>(values_above_pos_lambd, input_minus_bias);

    input_plus_bias = std::make_shared<v1::Multiply>(values_below_neg_lambd, input_plus_bias);

    return {std::make_shared<v1::Add>(input_plus_bias, input_minus_bias)};
}

ONNX_OP("Shrink", OPSET_SINCE(1), ai_onnx::opset_1::shrink);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
