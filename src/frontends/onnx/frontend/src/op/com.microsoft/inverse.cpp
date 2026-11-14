// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/inverse.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector inverse(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftinverse
    // Computes the inverse of every matrix in the batch: X has shape [..., N, N] and Y has the same shape.
    common::default_op_checks(node, 1, 1);

    const auto& x = node.get_ov_inputs()[0];
    const auto& x_type = x.get_element_type();
    CHECK_VALID_NODE(
        node,
        x_type.is_dynamic() || x_type == ov::element::f16 || x_type == ov::element::f32 || x_type == ov::element::f64,
        "Unsupported input data type for X, expected FP16, FP32 or FP64 but got: ",
        x_type);

    return {std::make_shared<v14::Inverse>(x, false)};
}

ONNX_OP("Inverse", OPSET_SINCE(1), com_microsoft::opset_1::inverse, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
