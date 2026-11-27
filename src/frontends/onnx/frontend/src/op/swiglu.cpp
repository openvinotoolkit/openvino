// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_28 {

ov::OutputVector swiglu(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://onnx.ai/onnx/operators/onnx__SwiGLU.html
    //   Y = Swish_alpha(A) * B, where Swish_alpha(a) = a * Sigmoid(alpha * a)
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node, inputs.size() == 2, "SwiGLU expects 2 input tensors (A, B). Got: ", inputs.size());

    const auto& gate = inputs.at(0);
    const auto& value = inputs.at(1);
    const auto alpha = node.get_attribute_as_constant<float>("alpha", 1.0f, gate.get_element_type());

    const auto swish = std::make_shared<v4::Swish>(gate, alpha);
    return {std::make_shared<v1::Multiply>(swish, value)};
}

// SwiGLU is a part of the standard ONNX domain since opset 28, but it is registered since opset 1,
// because OperatorsBridge requires a translator for every version imported by a model.
ONNX_OP("SwiGLU", OPSET_SINCE(1), ai_onnx::opset_28::swiglu);

}  // namespace ov::frontend::onnx::ai_onnx::opset_28
