// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_12 {
OutputVector softmax_cross_entropy_loss(const Node& node);
}  // namespace opset_12
namespace opset_13 {
OutputVector softmax_cross_entropy_loss(const Node& node);
}  // namespace opset_13
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov