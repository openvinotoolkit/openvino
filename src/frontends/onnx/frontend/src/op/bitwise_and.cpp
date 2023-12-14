// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/bitwise_and.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "default_opset.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector bitwise_and(const Node& node) {
    const auto inputs = node.get_ng_inputs();
    OPENVINO_ASSERT(inputs.size() == 2);
    return {std::make_shared<v13::BitwiseAnd>(inputs[0], inputs[1])};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
