// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs gather(const NodeContext& node) {
    const auto data_node = node.get_input("X");
    const auto index_node = node.get_input("Index");
    Output<Node> axis_node;

    if (node.has_input("Axis")) {
        axis_node = node.get_input("Axis");
    } else {
        const auto axis_value = node.get_attribute<int>("axis", 0);
        axis_node = default_opset::Constant::create(element::i32, Shape{}, {axis_value});
    }

    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Gather>(data_node, index_node, axis_node)},
        {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
