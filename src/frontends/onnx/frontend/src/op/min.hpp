// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"
#include "openvino/op/minimum.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline ov::OutputVector min(const ONNX_Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Minimum>(node, ov::op::AutoBroadcastType::NONE);
}

}  // namespace set_1

namespace set_8 {
inline ov::OutputVector min(const ONNX_Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Minimum>(node);
}

}  // namespace set_8

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
