// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector matmulintegertofloat(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 4);
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulintegertofloat
    const auto inputs = node.get_ov_inputs();
    const auto& a_int = inputs[0];
    const auto& b_int = inputs[1];
    const auto& a_scale = inputs[2];
    const auto& b_scale = inputs[3];

    ov::Output<ov::Node> a_zero_point =
        (inputs.size() > 4)
            ? inputs[4]
            : std::make_shared<ov::op::v0::Constant>(a_int.get_element_type(), ov::Shape{}, std::vector<int8_t>{0});
    ov::Output<ov::Node> b_zero_point =
        (inputs.size() > 5)
            ? inputs[5]
            : std::make_shared<ov::op::v0::Constant>(a_int.get_element_type(), ov::Shape{}, std::vector<int8_t>{0});

    CHECK_VALID_NODE(node,
                     a_int.get_element_type() == ov::element::i8 || a_int.get_element_type() == ov::element::u8,
                     "Unsupported input A type. Expected int8 or uint8, got: ",
                     a_int.get_element_type());

    CHECK_VALID_NODE(node,
                     b_int.get_element_type() == ov::element::i8 || b_int.get_element_type() == ov::element::u8,
                     "Unsupported input B type. Expected int8 or uint8, got: ",
                     b_int.get_element_type());

    const auto a_dequantized = std::make_shared<ov::op::v1::Subtract>(a_int, a_zero_point);
    const auto b_dequantized = std::make_shared<ov::op::v1::Subtract>(b_int, b_zero_point);

    const auto a_dequantized_converted =
        std::make_shared<ov::op::v0::Convert>(a_dequantized, a_scale.get_element_type());
    const auto b_dequantized_converted =
        std::make_shared<ov::op::v0::Convert>(b_dequantized, b_scale.get_element_type());

    const auto a_scaled = std::make_shared<ov::op::v1::Multiply>(a_dequantized_converted, a_scale);
    const auto b_scaled = std::make_shared<ov::op::v1::Multiply>(b_dequantized_converted, b_scale);

    const auto matmul_result = std::make_shared<ov::op::v0::MatMul>(a_scaled, b_scaled);

    if (inputs.size() > 6) {
        auto& bias = inputs[6];
        CHECK_VALID_NODE(node,
                         bias.get_partial_shape().rank().get_length() == 1,
                         "Bias tensor must be 1D. Got shape: ",
                         bias.get_partial_shape());

        const auto b_last_dim = b_int.get_partial_shape().get_shape().back();
        const auto bias_dim = bias.get_partial_shape()[0].get_length();
        CHECK_VALID_NODE(node,
                         static_cast<long int>(b_int.get_partial_shape().get_shape().back()) == bias_dim,
                         "Bias dimension must match the last dimension of B. Expected: ",
                         b_last_dim,
                         ", but got: ",
                         bias_dim);

        return {std::make_shared<ov::op::v1::Add>(matmul_result, bias)};
    }

    return {matmul_result};
}

ONNX_OP("MatMulIntegerToFloat", OPSET_SINCE(1), com_microsoft::opset_1::matmulintegertofloat, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov