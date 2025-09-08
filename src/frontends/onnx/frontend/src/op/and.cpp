// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/logical_and.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector logical_and(const ov::frontend::onnx::Node& node) {
    return common::handle_opset6_binary_op<ov::op::v1::LogicalAnd>(node);
}
ONNX_OP("And", OPSET_RANGE(1, 6), ai_onnx::opset_1::logical_and);
}  // namespace opset_1

namespace opset_7 {
ov::OutputVector logical_and(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::LogicalAnd>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}
ONNX_OP("And", OPSET_SINCE(7), ai_onnx::opset_7::logical_and);
}  // namespace opset_7
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
