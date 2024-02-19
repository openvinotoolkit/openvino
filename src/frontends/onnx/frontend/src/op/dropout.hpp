// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_12 {
ov::OutputVector dropout(const ov::frontend::onnx::Node& node);
}  // namespace set_12

namespace set_7 {
ov::OutputVector dropout(const ov::frontend::onnx::Node& node);
}  // namespace set_7

namespace set_1 {
ov::OutputVector dropout(const ov::frontend::onnx::Node& node);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
