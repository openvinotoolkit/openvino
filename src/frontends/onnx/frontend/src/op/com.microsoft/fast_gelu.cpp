// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector fast_gelu(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftfastgelu
    common::default_op_checks(node, 1);

    const auto inputs = node.get_ov_inputs();
    const auto& x = inputs[0];

    CHECK_VALID_NODE(node,
                     x.get_element_type() == ov::element::f16 || x.get_element_type() == ov::element::f32 ||
                         x.get_element_type() == ov::element::bf16,
                     "Unsupported input data type for X, expected FP16, FP32, or BF16 but got: ",
                     x.get_element_type());

    auto input_with_bias = x;

    if (inputs.size() > 1) {
        auto& bias = inputs[1];
        input_with_bias = std::make_shared<v1::Add>(x, bias);
    }

    const auto approximation_mode = ov::op::GeluApproximationMode::TANH;
    return {std::make_shared<v7::Gelu>(input_with_bias, approximation_mode)};
}

ONNX_OP("FastGelu", OPSET_SINCE(1), com_microsoft::opset_1::fast_gelu, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
