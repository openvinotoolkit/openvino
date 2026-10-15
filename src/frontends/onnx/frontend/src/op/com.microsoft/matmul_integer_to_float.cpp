// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
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

    ov::Output<ov::Node> a_zero_point;
    if (common::is_input_valid(node, 4)) {
        a_zero_point = inputs[4];
    }
    ov::Output<ov::Node> b_zero_point;
    if (common::is_input_valid(node, 5)) {
        b_zero_point = inputs[5];
    }

    CHECK_VALID_NODE(node,
                     a_int.get_element_type() == ov::element::i8 || a_int.get_element_type() == ov::element::u8,
                     "Unsupported input A type. Expected int8 or uint8, got: ",
                     a_int.get_element_type());

    CHECK_VALID_NODE(node,
                     b_int.get_element_type() == ov::element::i8 || b_int.get_element_type() == ov::element::u8,
                     "Unsupported input B type. Expected int8 or uint8, got: ",
                     b_int.get_element_type());

    auto a_scaled = ov::decomposition::low_precision_dequantize(a_int, a_scale, a_zero_point);
    auto b_scaled = ov::decomposition::low_precision_dequantize(b_int, b_scale, b_zero_point);

    const auto matmul_result = std::make_shared<ov::op::v0::MatMul>(a_scaled, b_scaled);

    if (common::is_input_valid(node, 6)) {
        auto& bias = inputs[6];
        const auto bias_shape = bias.get_partial_shape();
        CHECK_VALID_NODE(node,
                         bias_shape.rank().is_static() && bias_shape.rank().get_length() == 1,
                         "Bias tensor must be 1D. Got shape: ",
                         bias_shape);

        const auto b_shape = b_int.get_partial_shape();
        CHECK_VALID_NODE(node,
                         b_shape.rank().is_static() && b_shape.rank().get_length() >= 2,
                         "Input B must have static rank >= 2 to validate bias shape, got: ",
                         b_shape);
        const auto last_dim_idx = b_shape.rank().get_length() - 1;
        CHECK_VALID_NODE(node,
                         bias_shape[0].compatible(b_shape[last_dim_idx]),
                         "Bias dimension must match the last dimension of B. Expected: ",
                         b_shape[last_dim_idx],
                         ", but got: ",
                         bias_shape[0]);

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