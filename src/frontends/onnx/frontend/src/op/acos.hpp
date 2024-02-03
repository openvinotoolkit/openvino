// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "core/node.hpp"
#include "openvino/op/acos.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_7 {
inline ov::OutputVector acos(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Acos>(node.get_ng_inputs().at(0))};
}
}  // namespace set_7
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
OPENVINO_SUPPRESS_DEPRECATED_END
