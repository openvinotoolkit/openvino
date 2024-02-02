// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "core/node.hpp"
#include "openvino/op/acos.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_7 {
inline ov::OutputVector acos(const Node& node) {
    return {std::make_shared<ov::op::v0::Acos>(node.get_ng_inputs().at(0))};
}
}  // namespace set_7

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
