// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::frontend::onnx::ai_onnx {
namespace opset_12 {
OutputVector softmax_cross_entropy_loss(const Node& node);
}  // namespace opset_12
namespace opset_13 {
OutputVector softmax_cross_entropy_loss(const Node& node);
}  // namespace opset_13
}  // namespace ov::frontend::onnx::ai_onnx