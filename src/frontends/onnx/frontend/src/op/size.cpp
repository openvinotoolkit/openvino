// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/shape_of.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector size(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto axes = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto input_shape = std::make_shared<v3::ShapeOf>(data);
    return {std::make_shared<v1::ReduceProd>(input_shape, axes)};
}

ONNX_OP("Size", OPSET_SINCE(1), ai_onnx::opset_1::size);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
