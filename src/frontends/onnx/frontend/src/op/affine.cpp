// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector affine(const ov::frontend::onnx::Node& node) {
    // Affine is an obsolete experimental ONNX operation.
    // It takes one input tensor and produces one output tensor where
    // the affine function, y = alpha * x + beta, is applied to the input
    // elementwise.
    const auto inputs = node.get_ov_inputs();

    CHECK_VALID_NODE(node, inputs.size() == 1, "Affine expects 1 input tensor. Got: ", inputs.size());
    CHECK_VALID_NODE(node, node.has_attribute("alpha"), "\"alpha\" attribute is required.");
    CHECK_VALID_NODE(node, node.has_attribute("beta"), "\"beta\" attribute is required.");

    const auto data = inputs[0];
    const auto alpha_const = node.get_attribute_as_constant<float>("alpha", data.get_element_type());
    const auto beta_const = node.get_attribute_as_constant<float>("beta", data.get_element_type());

    return {std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(data, alpha_const), beta_const)};
}

ONNX_OP("Affine", OPSET_SINCE(1), ai_onnx::opset_1::affine);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
