// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
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
ov::OutputVector dynamic_quantize_matmul(const ov::frontend::onnx::Node& node) {
    // Original Documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.DynamicQuantizeMatMul

    // A, B, and b_scale are required inputs. b_zero_point and bias are optional inputs
    common::default_op_checks(node, 3);

    const auto inputs = node.get_ov_inputs();
    const auto& A = inputs[0];        // required
    const auto& B = inputs[1];        // required
    const auto& b_scale = inputs[2];  // required

    ov::Output<ov::Node> b_zero_point;  // optional, input[3]
    ov::Output<ov::Node> bias;          // optional, input[4]

    // Constrain input matrix A to T1 type (float tensor)
    auto element_type_A = A.get_element_type();
    CHECK_VALID_NODE(node,
                     element_type_A == ov::element::f32,
                     "Unsupported input A type, accepted FP32 but got: ",
                     element_type_A);

    // Constrain input matrix B to T2 type (int8 tensor, uint8 tensor)
    auto element_type_B = B.get_element_type();
    CHECK_VALID_NODE(node,
                     element_type_B == ov::element::u8 || element_type_B == ov::element::i8,
                     "Unsupported input B type, accepted UINT8, INT8 but got: ",
                     element_type_B);

    // Constrain input b_scale to T1 type (float tensor)
    auto element_type_b_scale = b_scale.get_element_type();
    CHECK_VALID_NODE(node,
                     element_type_b_scale == ov::element::f32,
                     "Unsupported input b_scale type, accepted FP32 but got: ",
                     element_type_b_scale);

    // Check for the optional inputs
    if (inputs.size() > 3) {
        // Constrain input b_zero_point to T2 type (int8 tensor, uint8 tensor)
        b_zero_point = inputs[3];
        auto element_type_b_zero_point = b_zero_point.get_element_type();
        CHECK_VALID_NODE(node,
                         element_type_b_zero_point == ov::element::u8 || element_type_b_zero_point == ov::element::i8,
                         "Unsupported input b_zero_point type, accepted UINT8, INT8 but got: ",
                         element_type_b_zero_point);
    }

    if (inputs.size() > 4) {
        // Constrain input bias to T1 type (float tensor)
        bias = inputs[4];
        auto element_type_bias = bias.get_element_type();
        CHECK_VALID_NODE(node,
                         element_type_bias == ov::element::f32,
                         "Unsupported input bias type, accepted FP32 but got: ",
                         element_type_bias);
    }

    // At time of writing, ov::MatMul does not support int8/uint8 types. To get the correct output, we need to
    // dequantize B. Technically this does not do DynamicQuantization, but is required for correct output of the
    // operator. It will implement A * B_dequantized + bias According to ONNX RT docs, they do linear quantization shown
    // here https://tomwildenhain-microsoft.github.io/onnxruntime/docs/performance/quantization.html B_dequantized = (B
    // - b_zero_point) * b_scale

    ov::Output<ov::Node> B_dequantized = std::make_shared<v0::Convert>(B, b_scale.get_element_type());
    b_zero_point = std::make_shared<v0::Convert>(b_zero_point, b_scale.get_element_type());
    B_dequantized = std::make_shared<v1::Subtract>(B_dequantized, b_zero_point);
    B_dequantized = std::make_shared<v1::Multiply>(B_dequantized, b_scale);

    // A, B are N-dimensional matrices. According to example ONNX models for this operator, the suboperations pass input
    // A/B such that B's shape is already transposed. E.g.
    // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/testdata/transform/fusion/dynamic_quantize_matmul.onnx
    // So here in ov::MatMul we will not do any transpose

    auto result = std::make_shared<v0::MatMul>(A, B_dequantized, false, false);

    // Adding bias if required
    if (bias.get_node_shared_ptr()) {
        return {std::make_shared<v1::Add>(result, bias)};
    }

    return {result};
}  // func end

ONNX_OP("DynamicQuantizeMatMul", OPSET_SINCE(1), com_microsoft::opset_1::dynamic_quantize_matmul, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
