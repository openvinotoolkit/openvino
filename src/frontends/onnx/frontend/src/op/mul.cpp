// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/multiply.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector mul(const ov::frontend::onnx::Node& node) {
    return common::handle_opset6_binary_op<ov::op::v1::Multiply>(node);
}

static bool registered = register_translator("Mul", VersionRange{1, 6}, ai_onnx::opset_1::mul);
}  // namespace opset_1

namespace opset_7 {
ov::OutputVector mul(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::Multiply>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

static bool registered = register_translator("Mul", VersionRange::since(7), ai_onnx::opset_7::mul);
}  // namespace opset_7
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
