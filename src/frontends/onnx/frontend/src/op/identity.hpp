// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector identity(const Node& node) {
    return node.get_ng_inputs();
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
