// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/size.hpp"

#include "openvino/core/shape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector size(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    auto axes = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto input_shape = std::make_shared<v3::ShapeOf>(data);
    return {std::make_shared<v1::ReduceProd>(input_shape, axes)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
