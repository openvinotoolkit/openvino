// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/op/add.hpp"
#include "utils/variadic.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
inline ov::OutputVector sum(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Add>(node, ov::op::AutoBroadcastType::NONE);
}

}  // namespace set_1

namespace set_8 {
inline ov::OutputVector sum(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Add>(node);
}

}  // namespace set_8
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
