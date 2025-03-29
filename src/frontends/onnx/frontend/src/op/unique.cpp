// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector unique(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const bool sorted = node.get_attribute_value<int64_t>("sorted", 1);

    if (node.has_attribute("axis")) {
        const auto axis = node.get_attribute_as_constant<int64_t>("axis");
        return std::make_shared<v10::Unique>(data, axis, sorted)->outputs();
    } else {
        return std::make_shared<v10::Unique>(data, sorted)->outputs();
    }
}
ONNX_OP("Unique", OPSET_SINCE(1), ai_onnx::opset_1::unique);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
