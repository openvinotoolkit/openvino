// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/equal.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector equal(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::Equal>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

ONNX_OP("Equal", OPSET_SINCE(1), ai_onnx::opset_1::equal);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
