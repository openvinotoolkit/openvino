// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/logical_or.hpp"

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector logical_or(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::LogicalOr>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

ONNX_OP("Or", OPSET_SINCE(1), ai_onnx::opset_1::logical_or);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
