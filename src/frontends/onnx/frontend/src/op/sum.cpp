// Copyright (C) 2018-2024 Intel Corporation
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

static bool registered = register_translator("Sum", VersionRange{1, 7}, sum);
}  // namespace opset_1

namespace opset_8 {
ov::OutputVector sum(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Add>(node);
}

static bool registered = register_translator("Sum", VersionRange::since(8), sum);
}  // namespace opset_8
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
