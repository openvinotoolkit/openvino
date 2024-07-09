// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/acosh.hpp"

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_9 {
ov::OutputVector acosh(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v3::Acosh>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Acosh", OPSET_SINCE(1), ai_onnx::opset_9::acosh);
}  // namespace opset_9
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
