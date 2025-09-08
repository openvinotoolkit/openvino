// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs partial_ops(const NodeContext& node, const std::string type) {
    auto x = node.get_ng_inputs("X");
    const auto start_index = node.get_attribute<int>("start_index");
    const auto length = node.get_attribute<int>("length");
    PADDLE_OP_CHECK(node, x[0].get_partial_shape().rank().get_length() == 2, "partial ops only support 2-D Tensor");

    int end_index;
    if (length < 0) {
        // Negative values for all elements after start_index on second dim.
        end_index = static_cast<int>(x[0].get_shape()[1]);
    } else {
        end_index = start_index + length;
    }

    const auto start_node = std::make_shared<default_opset::Constant>(element::i32, Shape{1}, start_index);
    const auto end_node = std::make_shared<default_opset::Constant>(element::i32, Shape{1}, end_index);
    const auto one_node = std::make_shared<default_opset::Constant>(element::i32, Shape{1}, 1);

    auto left = std::make_shared<default_opset::Slice>(x[0], start_node, end_node, one_node, one_node);
    auto right = std::make_shared<default_opset::Slice>(x[1], start_node, end_node, one_node, one_node);

    if (type == "sum")
        return node.default_single_output_mapping({std::make_shared<default_opset::Add>(left, right)}, {"Out"});
    return node.default_single_output_mapping({std::make_shared<default_opset::Concat>(NodeVector{left, right}, 1)},
                                              {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
