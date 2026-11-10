// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/less.hpp"

#include "core/operator_set.hpp"

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector less(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::Less>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

ONNX_OP("Less", OPSET_SINCE(1), ai_onnx::opset_1::less);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
