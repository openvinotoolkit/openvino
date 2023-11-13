// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include <memory>

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector add(const Node& node);

}  // namespace set_1

namespace set_6 {
OutputVector add(const Node& node);

}  // namespace set_6

namespace set_7 {
OutputVector add(const Node& node);

}  // namespace set_7

namespace set_13 {
using set_7::add;
}  // namespace set_13

namespace set_14 {
using set_13::add;
}  // namespace set_14

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
