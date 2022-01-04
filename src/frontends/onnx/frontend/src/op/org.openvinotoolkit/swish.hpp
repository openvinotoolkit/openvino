// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector swish(const Node& node);
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
