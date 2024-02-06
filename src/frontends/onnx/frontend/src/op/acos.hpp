// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/op/acos.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_7 {
inline ov::OutputVector acos(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Acos>(node.get_ov_inputs().at(0))};
}
}  // namespace set_7
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
