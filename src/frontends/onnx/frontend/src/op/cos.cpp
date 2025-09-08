// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cos.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector cos(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v0::Cos>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Cos", OPSET_SINCE(1), ai_onnx::opset_1::cos);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
