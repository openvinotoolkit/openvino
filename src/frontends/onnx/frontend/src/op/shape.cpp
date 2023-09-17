// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/shape.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"
#include "op/shape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector shape(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    return {std::make_shared<default_opset::ShapeOf>(data)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
