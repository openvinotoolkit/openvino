// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sin.hpp"

#include "core/operator_set.hpp"

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector sin(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Sin>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Sin", OPSET_SINCE(1), ai_onnx::opset_1::sin);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
