// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/maximum.hpp"
#include "utils/variadic.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector max(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Maximum>(node, ov::op::AutoBroadcastType::NONE);
}

ONNX_OP("Max", OPSET_RANGE(1, 7), ai_onnx::opset_1::max);
}  // namespace opset_1

namespace opset_8 {
ov::OutputVector max(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Maximum>(node);
}

ONNX_OP("Max", OPSET_SINCE(8), ai_onnx::opset_8::max);
}  // namespace opset_8
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
