// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atan.hpp"

#include "core/operator_set.hpp"

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector atan(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Atan>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Atan", OPSET_SINCE(1), ai_onnx::opset_1::atan);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
