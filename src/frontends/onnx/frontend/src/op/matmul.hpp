// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "onnx_import/core/node.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace detail {
inline OutputVector matmul(const Output<ov::Node>& a, const Output<ov::Node>& b) {
    return {std::make_shared<default_opset::MatMul>(a, b)};
}
}  // namespace detail
namespace set_1 {
inline OutputVector matmul(const Node& node) {
    return {std::make_shared<default_opset::MatMul>(node.get_ng_inputs().at(0), node.get_ng_inputs().at(1))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ov
