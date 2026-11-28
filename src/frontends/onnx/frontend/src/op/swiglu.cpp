// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_28 {

ov::OutputVector swiglu(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://onnx.ai/onnx/operators/onnx__SwiGLU.html
    //   Y = Swish(A, alpha) * B, where Swish(a, alpha) = a * Sigmoid(alpha * a)
    common::default_op_checks(node, 2, 2);
    const auto inputs = node.get_ov_inputs();
    const auto& gate = inputs[0];
    const auto& value = inputs[1];

    CHECK_VALID_NODE(node,
                     gate.get_partial_shape().compatible(value.get_partial_shape()),
                     "SwiGLU expects inputs A and B to have identical shapes, got: ",
                     gate.get_partial_shape(),
                     " and ",
                     value.get_partial_shape());

    const auto alpha = node.get_attribute_value<float>("alpha", 1.0f);
    std::shared_ptr<ov::Node> swish;
    if (alpha == 1.0f) {
        // Swish without beta (SiLU) is the form matched by plugin fusions, e.g. GLUFusion.
        swish = std::make_shared<v4::Swish>(gate);
    } else {
        ov::Output<ov::Node> beta = node.get_attribute_as_constant<float>("alpha", alpha, gate.get_element_type());
        if (gate.get_element_type().is_dynamic()) {
            beta = std::make_shared<v1::ConvertLike>(beta, gate);
        }
        swish = std::make_shared<v4::Swish>(gate, beta);
    }
    return {std::make_shared<v1::Multiply>(swish, value)};
}

// SwiGLU is a part of the standard ONNX domain since opset 28, but it is registered since opset 1,
// because OperatorsBridge requires a translator for every version imported by a model.
ONNX_OP("SwiGLU", OPSET_SINCE(1), ai_onnx::opset_28::swiglu);

}  // namespace ov::frontend::onnx::ai_onnx::opset_28
