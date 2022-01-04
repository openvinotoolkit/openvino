// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "openvino/core/node_vector.hpp"
#include "onnx_import/core/node.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector identity(const Node& node) {
    return node.get_ng_inputs();
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
