// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/operator_set.hpp"
#include "openvino/op/divide.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector div(const ov::frontend::onnx::Node& node) {
    return common::handle_opset6_binary_op<ov::op::v1::Divide>(node);
}

static bool registered = register_translator("Div", VersionRange{1, 6}, div);
}  // namespace set_1

namespace set_7 {
ov::OutputVector div(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::Divide>(node.get_ov_inputs().at(0), node.get_ov_inputs().at(1))};
}

static bool registered = register_translator("Div", VersionRange::since(7), div);
}  // namespace set_7
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
