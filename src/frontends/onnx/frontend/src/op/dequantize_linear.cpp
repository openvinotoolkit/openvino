// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

namespace {
ov::Output<ov::Node> get_zero_point(const NodeContext& node) {
    if (node.has_input("ZeroPoint")) {
        return node.get_input("ZeroPoint");
    } else {
        return std::make_shared<default_opset::Constant>(ov::element::i32, ov::Shape{1}, 0);
    }
}

ov::Output<ov::Node> reshape_for_broadcast(const ov::Output<ov::Node>& input, int64_t axis, const ov::Shape& x_shape) {
    if (input.get_partial_shape().rank().get_length() == 0) {
        return input;
    }

    ov::Shape target_shape(x_shape.size(), 1);
    target_shape[axis] = input.get_shape()[0];

    auto shape_const =
        std::make_shared<default_opset::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
    return std::make_shared<default_opset::Reshape>(input, shape_const, true);
}

}  // namespace

NamedOutputs dequantize_linear(const NodeContext& node) {
    auto x = node.get_input("X");
    auto y_scale = node.get_input("Scale");
    auto y_zero_point = get_zero_point(node);
    auto axis = node.get_attribute<int64_t>("axis", 1);

    const auto& x_shape = x.get_partial_shape();
    PADDLE_OP_CHECK(node, x_shape.rank().is_static(), "Rank of input tensor must be static");
    //A simplified version of normalize_axis
    axis =  axis < 0 ? axis + x_shape.rank().get_length() : axis;

    const auto& input_type = x.get_element_type();
    const auto& output_type = ov::element::f32;

    y_scale = reshape_for_broadcast(y_scale, axis, x_shape.get_shape());
    y_zero_point = reshape_for_broadcast(y_zero_point, axis, x_shape.get_shape());

    auto zero_point = std::make_shared<default_opset::Convert>(y_zero_point, input_type);
    auto scale = std::make_shared<default_opset::Convert>(y_scale, output_type);

    // Dequantization formula: (x - zero_point) * scale
    auto dequantized = std::make_shared<default_opset::Multiply>(
        std::make_shared<default_opset::Subtract>(std::make_shared<default_opset::Convert>(x, output_type),
                                                  std::make_shared<default_opset::Convert>(zero_point, output_type)),
        scale);

    return node.default_single_output_mapping({dequantized}, {"Y"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov