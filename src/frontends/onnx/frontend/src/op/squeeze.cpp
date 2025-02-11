// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector squeeze(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    const auto axes = node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

    if (axes.empty()) {
        return {std::make_shared<v0::Squeeze>(data)};
    } else {
        const auto axes_const = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
        return {std::make_shared<v0::Squeeze>(data, axes_const)};
    }
}

ONNX_OP("Squeeze", OPSET_RANGE(1, 12), ai_onnx::opset_1::squeeze);
}  // namespace opset_1

namespace opset_13 {
ov::OutputVector squeeze(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    if (inputs.size() < 2) {
        return {std::make_shared<v0::Squeeze>(inputs.at(0))};
    } else {
        return {std::make_shared<v0::Squeeze>(inputs.at(0), inputs.at(1))};
    }
}

ONNX_OP("Squeeze", OPSET_SINCE(13), ai_onnx::opset_13::squeeze);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
