// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector bias_add(const ov::frontend::onnx::Node& node) {
    // Documentation: BiasAdd computes Y = X + bias + skip
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftbiasadd

    common::default_op_checks(node, 3);

    const auto inputs = node.get_ov_inputs();
    const auto& X = inputs[0];
    const auto& bias = inputs[1];
    const auto& skip = inputs[2];

    auto element_type = X.get_element_type();
    CHECK_VALID_NODE(node,
                     element_type == ov::element::f16 || element_type == ov::element::f32,
                     "Unsupported input data type for X, expected FP16 or FP32 but got: ",
                     element_type);

    const auto& X_shape = X.get_partial_shape();
    const auto& bias_shape = bias.get_partial_shape();
    const auto& skip_shape = skip.get_partial_shape();

    CHECK_VALID_NODE(node,
                     X_shape.rank().is_static() && X_shape.rank().get_length() == 3,
                     "Input X must have rank 3 (N, S, C), but got: ",
                     X_shape);
    CHECK_VALID_NODE(node,
                     skip_shape.rank().is_static() && skip_shape.rank().get_length() == 3,
                     "Input skip must have rank 3 (N, S, C), but got: ",
                     skip_shape);

    CHECK_VALID_NODE(node,
                     X_shape.compatible(skip_shape),
                     "Input X and skip must have the same shape, but got: X=",
                     X_shape,
                     ", skip=",
                     skip_shape);

    CHECK_VALID_NODE(node,
                     bias_shape.rank().is_static() && bias_shape.rank().get_length() == 1,
                     "Input bias must have rank 1 (C), but got: ",
                     bias_shape);

    CHECK_VALID_NODE(node,
                     X_shape[2].compatible(bias_shape[0]),
                     "Input bias shape must match the channel dimension (C) of X and skip, but got: bias=",
                     bias_shape,
                     ", X=",
                     X_shape);

    auto X_plus_bias = std::make_shared<v1::Add>(X, bias);

    return {std::make_shared<v1::Add>(X_plus_bias, skip)};
}

ONNX_OP("BiasAdd", OPSET_SINCE(1), com_microsoft::opset_1::bias_add, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
