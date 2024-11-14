// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs roll(const NodeContext& node) {
    auto input_node = node.get_input("X");
    Output<Node> shifts_node;
    if (node.has_input("ShiftsTensor")) {
        auto shifts = node.get_input("ShiftsTensor");
        auto shifts_var_node = std::make_shared<default_opset::Convert>(shifts, element::i64);
        shifts_node = std::make_shared<default_opset::Squeeze>(shifts_var_node);
    } else {
        const auto shifts = node.get_attribute<std::vector<int64_t>>("shifts");
        shifts_node = default_opset::Constant::create(element::i64, {shifts.size()}, shifts);
    }

    std::vector<int64_t> axis = node.get_attribute<std::vector<int64_t>>("axis");
    if (axis.empty()) {
        const auto const_minus_1 = default_opset::Constant::create(element::i64, Shape{1}, {-1});
        Output<Node> axis_node = default_opset::Constant::create(element::i64, Shape{1}, {0});
        const auto flat = std::make_shared<default_opset::Reshape>(input_node, const_minus_1, false);
        const auto roll = std::make_shared<default_opset::Roll>(flat, shifts_node, axis_node);
        const auto shape_of_data = std::make_shared<default_opset::ShapeOf>(input_node, element::i64);
        const auto reshape = std::make_shared<default_opset::Reshape>(roll, shape_of_data, false);
        return node.default_single_output_mapping(reshape, {"Out"});
    } else {
        Output<Node> axis_node = default_opset::Constant::create(element::i64, {axis.size()}, axis);
        return node.default_single_output_mapping(
            std::make_shared<default_opset::Roll>(input_node, shifts_node, axis_node),
            {"Out"});
    }
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
