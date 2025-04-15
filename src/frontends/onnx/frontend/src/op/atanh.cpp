// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atanh.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector atanh(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v3::Atanh>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Atanh", OPSET_SINCE(1), ai_onnx::opset_1::atanh);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
