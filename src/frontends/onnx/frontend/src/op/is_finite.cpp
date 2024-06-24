// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/opsets/opset10.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector is_finite(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v10::IsFinite>(data)};
}

ONNX_OP("IsFinite", OPSET_SINCE(1), ai_onnx::opset_1::is_finite);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
