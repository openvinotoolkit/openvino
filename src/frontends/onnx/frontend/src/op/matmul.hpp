// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "openvino/op/matmul.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
inline OutputVector matmul(const Output<ov::Node>& a, const Output<ov::Node>& b) {
    return {std::make_shared<ov::op::v0::MatMul>(a, b)};
}
}  // namespace detail
namespace set_1 {
inline OutputVector matmul(const Node& node) {
    return {std::make_shared<ov::op::v0::MatMul>(node.get_ng_inputs().at(0), node.get_ng_inputs().at(1))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
