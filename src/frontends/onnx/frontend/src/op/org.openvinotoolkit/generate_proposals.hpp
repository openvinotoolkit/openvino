// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector generate_proposals(const Node& node);
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
