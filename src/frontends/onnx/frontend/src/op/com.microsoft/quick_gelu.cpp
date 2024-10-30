// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector quick_gelu(const ov::frontend::onnx::Node& node) {
    // Original Documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QuickGelu
    // Goal: Compute x * Sigmoid(alpha * x)
    common::default_op_checks(node, 1);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];

    // Constrain input type to float16, float, double (f64), bfloat16
    auto element_type = x.get_element_type();
    CHECK_VALID_NODE(node,
                     element_type == ov::element::f16 || element_type == ov::element::f32 ||
                         element_type == ov::element::f64 || element_type == ov::element::bf16,
                     "Unsupported input x type, accepted FP16, FP32, FP64, BFP16 but got: ",
                     element_type);

    // Get attribute from node
    const float alpha = node.get_attribute_value<float>("alpha");

    // Numpy broadcasting rule is automatically applied with mismatched shapes according to:
    // https://docs.openvino.ai/2022.3/openvino_docs_ops_arithmetic_Multiply_1.html "Tensor with dimension of size 1
    // will be implicitly broadcasted to match the size of the second tensor." Convert alpha to tensor with size 1
    const auto alpha_tensor = std::make_shared<v0::Constant>(ov::element::f32, Shape{1}, alpha);

    auto alpha_x = std::make_shared<v1::Multiply>(alpha_tensor, x);
    auto sig_alpha_x = std::make_shared<v0::Sigmoid>(alpha_x);
    auto result = std::make_shared<v1::Multiply>(x, sig_alpha_x);

    return {result};
}  // func end

ONNX_OP("QuickGelu", OPSET_SINCE(1), com_microsoft::opset_1::quick_gelu, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
