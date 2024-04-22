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
ov::OutputVector add(const ov::frontend::onnx::Node& node);

}  // namespace set_1

namespace set_6 {
ov::OutputVector add(const ov::frontend::onnx::Node& node);

}  // namespace set_6

namespace set_7 {
ov::OutputVector add(const ov::frontend::onnx::Node& node);

}  // namespace set_7

namespace set_13 {
using set_7::add;
}  // namespace set_13

namespace set_14 {
using set_13::add;
}  // namespace set_14
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
