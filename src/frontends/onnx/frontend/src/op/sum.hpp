// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "core/node.hpp"
#include "openvino/op/add.hpp"
#include "utils/variadic.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline ov::OutputVector sum(const Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Add>(node, ov::op::AutoBroadcastType::NONE);
}

}  // namespace set_1

namespace set_8 {
inline ov::OutputVector sum(const Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Add>(node);
}

}  // namespace set_8

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
