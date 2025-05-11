// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/minimum.hpp"
#include "utils/variadic.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector min(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Minimum>(node, ov::op::AutoBroadcastType::NONE);
}

ONNX_OP("Min", {1, 7}, ai_onnx::opset_1::min);
}  // namespace opset_1

namespace opset_8 {
ov::OutputVector min(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Minimum>(node);
}

ONNX_OP("Min", OPSET_SINCE(8), ai_onnx::opset_8::min);
}  // namespace opset_8
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
