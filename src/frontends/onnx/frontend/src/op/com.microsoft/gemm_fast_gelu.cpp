// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector gemm_fast_gelu(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgemmfastgelu
    // Y = FastGelu(X @ W + bias): a fusion of MatMul and FastGelu (tanh approximation of Gelu).
    common::default_op_checks(node, 2, 3);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& x_type = x.get_element_type();
    CHECK_VALID_NODE(
        node,
        x_type.is_dynamic() || x_type == ov::element::f16 || x_type == ov::element::f32 || x_type == ov::element::bf16,
        "Unsupported input data type for X, expected FP16, FP32 or BF16 but got: ",
        x_type);

    ov::Output<ov::Node> result = std::make_shared<v0::MatMul>(x, w);
    if (common::is_input_valid(inputs, 2)) {
        result = std::make_shared<v1::Add>(result, inputs[2]);
    }

    return {std::make_shared<v7::Gelu>(result, ov::op::GeluApproximationMode::TANH)};
}

ONNX_OP("GemmFastGelu", OPSET_SINCE(1), com_microsoft::opset_1::gemm_fast_gelu, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
