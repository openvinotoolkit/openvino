// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/add.hpp"
#include "utils/variadic.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector sum(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Add>(node, ov::op::AutoBroadcastType::NONE);
}

ONNX_OP("Sum", OPSET_RANGE(1, 7), ai_onnx::opset_1::sum);
}  // namespace opset_1

namespace opset_8 {
ov::OutputVector sum(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Add>(node);
}

ONNX_OP("Sum", OPSET_SINCE(8), ai_onnx::opset_8::sum);
}  // namespace opset_8
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
