// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs flip(const NodeContext& node) {
    const auto data_node = node.get_input("X");
    const auto axes = node.get_attribute<std::vector<int32_t>>("axis");
    const auto input_shape = data_node.get_partial_shape().get_shape();
    const auto dims = static_cast<int32_t>(data_node.get_partial_shape().rank().get_length());
    const auto dtype = data_node.get_element_type();
    // for zero-dim input
    PADDLE_OP_CHECK(node, (dims > 0), "Input dims must be greater than 0");

    Output<Node> temp = data_node;
    std::vector<Output<Node>> temp_split_out;
    int32_t axis;
    for (size_t idx = 0; idx < axes.size(); idx++) {
        axis = axes[idx];
        if (axis < 0)
            axis += dims;
        // Do nothing when dims of selected axis are 1.
        if (input_shape[axis] != 1) {
            const auto split_axes = default_opset::Constant::create(element::i64, Shape{}, {axis});
            auto split = std::make_shared<default_opset::Split>(temp, split_axes, input_shape[axis]);
            temp_split_out = split->outputs();
            // reverse the vector<Node> then concat
            std::reverse(temp_split_out.begin(), temp_split_out.end());
            temp = std::make_shared<default_opset::Concat>(temp_split_out, axis);
        }
    }
    // for output, convert Output<Node> to shared_ptr<Node>
    return node.default_single_output_mapping({std::make_shared<default_opset::Convert>(temp, dtype)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
