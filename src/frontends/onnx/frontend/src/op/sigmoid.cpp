// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sigmoid.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector sigmoid(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Sigmoid>(node.get_ov_inputs().at(0))};
}

ONNX_OP("Sigmoid", OPSET_SINCE(1), ai_onnx::opset_1::sigmoid);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
