// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"
#include "openvino/op/logical_or.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline ov::OutputVector logical_or(const ONNX_Node& node) {
    return {std::make_shared<ov::op::v1::LogicalOr>(node.get_ng_inputs().at(0), node.get_ng_inputs().at(1))};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
