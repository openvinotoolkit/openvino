// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/logical_xor.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector logical_xor(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::LogicalXor>(node.get_ov_inputs().at(0),
                                                     node.get_ov_inputs().at(1),
                                                     ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))};
}

ONNX_OP("Xor", OPSET_SINCE(1), ai_onnx::opset_1::logical_xor);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
