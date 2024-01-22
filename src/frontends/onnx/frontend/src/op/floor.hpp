// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"
#include "openvino/op/floor.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector floor(const Node& node) {
    return {std::make_shared<ov::op::v0::Floor>(node.get_ng_inputs().at(0))};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
