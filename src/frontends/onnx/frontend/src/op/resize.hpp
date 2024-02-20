// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector resize(const ov::frontend::onnx::Node& node);

}  // namespace set_1

namespace set_11 {
ov::OutputVector resize(const ov::frontend::onnx::Node& node);
}

}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
