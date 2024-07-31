// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sqrt.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector sqrt(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Sqrt>(node.get_ov_inputs().at(0))};
}

ONNX_OP("Sqrt", OPSET_SINCE(1), ai_onnx::opset_1::sqrt);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
