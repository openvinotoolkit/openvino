// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector gated_add(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftgatedadd
    // Plain mode:     output = X + Y * gate, where gate has shape (..., 1).
    // Projected mode: output = X + Y * Sigmoid(gate @ gate_weight), where gate has shape (..., K) and the
    //                 optional gate_weight has shape (K, 1).
    common::default_op_checks(node, 3, 4);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];
    const auto& y = inputs[1];
    const auto& gate = inputs[2];
    const auto& x_type = x.get_element_type();
    CHECK_VALID_NODE(
        node,
        x_type.is_dynamic() || x_type == ov::element::f16 || x_type == ov::element::f32 || x_type == ov::element::bf16,
        "Unsupported input data type for X, expected FP16, FP32 or BF16 but got: ",
        x_type);

    ov::Output<ov::Node> effective_gate = gate;
    if (common::is_input_valid(inputs, 3)) {
        const auto projected_gate = std::make_shared<v0::MatMul>(gate, inputs[3]);
        effective_gate = std::make_shared<v0::Sigmoid>(projected_gate);
    }

    const auto scaled_y = std::make_shared<v1::Multiply>(y, effective_gate);
    return {std::make_shared<v1::Add>(x, scaled_y)};
}

ONNX_OP("GatedAdd", OPSET_SINCE(1), com_microsoft::opset_1::gated_add, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
