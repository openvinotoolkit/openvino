// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector unsqueeze(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto axes_node = node.get_attribute_as_constant<std::vector<std::int64_t>>("axes", {});
    return {std::make_shared<v0::Unsqueeze>(data, axes_node)};
}

ONNX_OP("Unsqueeze", OPSET_RANGE(1, 12), ai_onnx::opset_1::unsqueeze);
}  // namespace opset_1

namespace opset_13 {
ov::OutputVector unsqueeze(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    return {std::make_shared<v0::Unsqueeze>(inputs.at(0), inputs.at(1))};
}

ONNX_OP("Unsqueeze", OPSET_SINCE(13), ai_onnx::opset_13::unsqueeze);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
