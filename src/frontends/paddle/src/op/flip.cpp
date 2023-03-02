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
    auto input_shape = data_node.get_partial_shape();
    PADDLE_OP_CHECK(node, (input_shape.rank().is_static()), "flip not support dynamic rank!");
    const auto dims = static_cast<int32_t>(data_node.get_partial_shape().rank().get_length());
    const auto dtype = data_node.get_element_type();
    bool is_1dim = false;
    int32_t axis, shape_len, batch_index;
    Output<Node> temp = data_node;
    const auto uns_axes = default_opset::Constant::create(ov::element::i64, {1}, {1});
    if (dims == 1) {
        temp = std::make_shared<default_opset::Unsqueeze>(temp, uns_axes);
        input_shape = temp.get_partial_shape();
        is_1dim = true;
    }
    for (size_t idx = 0; idx < axes.size(); idx++) {
        axis = axes[idx];
        axis += axis < 0 ? dims : 0;
        shape_len = input_shape[axis].get_length();
        batch_index = axis == 0 ? 1 : 0;
        auto batch_len = input_shape[batch_index].get_length();
        const auto seq_length =
            default_opset::Constant::create(element::i64, {static_cast<uint32_t>(batch_len)}, {shape_len});
        temp = std::make_shared<default_opset::ReverseSequence>(temp, seq_length, batch_index, axis);
    }
    if (is_1dim)
        return node.default_single_output_mapping({std::make_shared<default_opset::Squeeze>(temp, uns_axes)}, {"Out"});
    // for output, convert Output<Node> to shared_ptr<Node>
    return node.default_single_output_mapping({std::make_shared<default_opset::Convert>(temp, dtype)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov