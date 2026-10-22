// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erf.hpp"

#include "core/operator_set.hpp"

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector erf(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Erf>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Erf", OPSET_SINCE(1), ai_onnx::opset_1::erf);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
