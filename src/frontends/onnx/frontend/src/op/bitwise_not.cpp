// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_not.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector bitwise_not(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    OPENVINO_ASSERT(inputs.size() == 1);
    return {std::make_shared<v13::BitwiseNot>(inputs[0])};
}
ONNX_OP("BitwiseNot", OPSET_SINCE(1), ai_onnx::opset_1::bitwise_not);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
