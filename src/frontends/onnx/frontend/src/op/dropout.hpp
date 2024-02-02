// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_12 {
ov::OutputVector dropout(const Node& node);
}  // namespace set_12

namespace set_7 {
ov::OutputVector dropout(const Node& node);
}  // namespace set_7

namespace set_1 {
ov::OutputVector dropout(const Node& node);
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
