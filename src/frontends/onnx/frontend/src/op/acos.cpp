// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/acos.hpp"

#include "core/operator_set.hpp"

namespace ov::frontend::onnx::ai_onnx::opset_7 {
ov::OutputVector acos(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Acos>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Acos", OPSET_SINCE(1), ai_onnx::opset_7::acos);
}  // namespace ov::frontend::onnx::ai_onnx::opset_7
