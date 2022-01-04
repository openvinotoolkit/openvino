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
namespace set_1 {
inline OutputVector sign(const Node& node) {
    return {std::make_shared<default_opset::Sign>(node.get_ng_inputs().at(0))};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
