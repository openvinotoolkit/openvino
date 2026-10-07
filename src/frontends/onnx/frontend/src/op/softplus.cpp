// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softplus.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector softplus(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v4::SoftPlus>(data)};
}

ONNX_OP("Softplus", OPSET_SINCE(1), ai_onnx::opset_1::softplus);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
