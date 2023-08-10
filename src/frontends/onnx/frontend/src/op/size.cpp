// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/size.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector size(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    auto axes = default_opset::Constant::create(ngraph::element::i32, Shape{}, {0});
    auto input_shape = std::make_shared<default_opset::ShapeOf>(data);
    return {std::make_shared<default_opset::ReduceProd>(input_shape, axes)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
