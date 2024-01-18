// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/bitwise_not.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "default_opset.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector bitwise_not(const Node& node) {
    return {std::make_shared<v13::BitwiseNot>(node.get_ng_inputs().at(0))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
