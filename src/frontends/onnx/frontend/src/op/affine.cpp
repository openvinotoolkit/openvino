// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/affine.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/shape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector affine(const Node& node) {
    // Affine is an obsolete experimental ONNX operation.
    // It takes one input tensor and produces one output tensor where
    // the affine function, y = alpha * x + beta, is applied to the input
    // elementwise.
    const auto inputs = node.get_ng_inputs();

    CHECK_VALID_NODE(node, inputs.size() == 1, "Affine expects 1 input tensor. Got: ", inputs.size());
    CHECK_VALID_NODE(node, node.has_attribute("alpha"), "\"alpha\" attribute is required.");
    CHECK_VALID_NODE(node, node.has_attribute("beta"), "\"beta\" attribute is required.");

    const auto data = inputs[0];
    const auto alpha_const = node.get_attribute_as_constant<float>("alpha", data.get_element_type());
    const auto beta_const = node.get_attribute_as_constant<float>("beta", data.get_element_type());

    return {
        std::make_shared<default_opset::Add>(std::make_shared<default_opset::Multiply>(data, alpha_const), beta_const)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
