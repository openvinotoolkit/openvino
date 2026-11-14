// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft::opset_1 {

ov::OutputVector scaled_silu(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftscaledsilu
    // Z = scale * X, Y = Z * Sigmoid(Z), where scale is the optional scalar input or the alpha attribute.
    // Z * Sigmoid(Z) is Swish(Z) with beta = 1.
    common::default_op_checks(node, 1, 2);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];
    const auto& x_type = x.get_element_type();
    CHECK_VALID_NODE(
        node,
        x_type.is_dynamic() || x_type == ov::element::f16 || x_type == ov::element::f32 || x_type == ov::element::bf16,
        "Unsupported input data type for X, expected FP16, FP32 or BF16 but got: ",
        x_type);

    ov::Output<ov::Node> scale;
    if (common::is_input_valid(inputs, 1)) {
        scale = ov::frontend::onnx::reshape::interpret_as_scalar(inputs[1]);
        if (scale.get_element_type() != x_type) {
            scale = std::make_shared<v1::ConvertLike>(scale, x);
        }
    } else {
        scale = node.get_attribute_as_constant<float>("alpha", 1.0f, x_type);
    }

    const auto scaled_x = std::make_shared<v1::Multiply>(x, scale);
    return {std::make_shared<v4::Swish>(scaled_x)};
}

ONNX_OP("ScaledSiLU", OPSET_SINCE(1), com_microsoft::opset_1::scaled_silu, MICROSOFT_DOMAIN);

}  // namespace ov::frontend::onnx::com_microsoft::opset_1
