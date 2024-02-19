// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/op/sign.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
inline ov::OutputVector sign(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v0::Sign>(node.get_ov_inputs().at(0))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
