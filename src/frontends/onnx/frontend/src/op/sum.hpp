// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/variadic.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector sum(const Node& node) {
    return variadic::make_ng_variadic_op<default_opset::Add>(node, ngraph::op::AutoBroadcastType::NONE);
}

}  // namespace set_1

namespace set_8 {
inline OutputVector sum(const Node& node) {
    return variadic::make_ng_variadic_op<default_opset::Add>(node);
}

}  // namespace set_8

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
