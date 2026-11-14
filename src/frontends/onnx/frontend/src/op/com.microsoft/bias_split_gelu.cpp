// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector bias_split_gelu(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftbiassplitgelu
    // Y = left_half(X + bias) * Gelu(right_half(X + bias)), where the halves are split along the last axis.
    common::default_op_checks(node, 2, 2);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];
    const auto& bias = inputs[1];
    const auto& x_type = x.get_element_type();
    CHECK_VALID_NODE(node,
                     x_type.is_dynamic() || x_type == ov::element::f16 || x_type == ov::element::f32,
                     "Unsupported input data type for X, expected FP16 or FP32 but got: ",
                     x_type);

    const auto x_with_bias = std::make_shared<v1::Add>(x, bias);
    const auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    const auto halves = std::make_shared<v1::Split>(x_with_bias, split_axis, 2);
    const auto gelu = std::make_shared<v7::Gelu>(halves->output(1), ov::op::GeluApproximationMode::ERF);

    return {std::make_shared<v1::Multiply>(halves->output(0), gelu)};
}

ONNX_OP("BiasSplitGelu", OPSET_SINCE(1), com_microsoft::opset_1::bias_split_gelu, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
