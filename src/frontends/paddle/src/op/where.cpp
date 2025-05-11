// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs where(const NodeContext& node) {
    const auto condition_node = node.get_input("Condition");
    const auto x_node = node.get_input("X");
    const auto y_node = node.get_input("Y");
    // TODO: support 'shape x != shape y' #83233
    const auto x_shape = x_node.get_partial_shape();
    const auto y_shape = y_node.get_partial_shape();
    PADDLE_OP_CHECK(node, x_shape.compatible(y_shape), "shape x should be compatible to shape y!");

    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Select>(condition_node, x_node, y_node, ov::op::AutoBroadcastType::PDPD)},
        {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
