// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cosh.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector cosh(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v0::Cosh>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Cosh", OPSET_SINCE(1), ai_onnx::opset_1::cosh);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
