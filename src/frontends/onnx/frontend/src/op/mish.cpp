// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/mish.hpp"

#include <memory>
#include <vector>

#include "default_opset.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector mish(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    return OutputVector{std::make_shared<default_opset::Mish>(data)};
}

}  // namespace set_1 

}  // namespace op 

}  // namespace onnx_import 

}  // namespace ngraph 
OPENVINO_SUPPRESS_DEPRECATED_END
