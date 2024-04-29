// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/op/logical_and.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
inline ov::OutputVector logical_and(const ov::frontend::onnx::Node& node) {
    return common::handle_opset6_binary_op<ov::op::v1::LogicalAnd>(node);
}
}  // namespace set_1

namespace set_7 {
inline ov::OutputVector logical_and(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::LogicalAnd>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}
}  // namespace set_7
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
