// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/negative.hpp"

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector neg(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Negative>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Neg", OPSET_SINCE(1), ai_onnx::opset_1::neg);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
