// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs tril_triu(const NodeContext& node) {
    auto x = node.get_input("X");
    const int diagonal = node.get_attribute<int>("diagonal");
    const auto lower = node.get_attribute<bool>("lower");
    PADDLE_OP_CHECK(node, x.get_partial_shape().rank().get_length() == 2, "partial ops only support 2-D Tensor");

    const auto diag_node = default_opset::Constant::create(element::i32, Shape{}, {diagonal});
    const auto zero_node = default_opset::Constant::create(element::i32, Shape{}, {0});
    const auto one_node = default_opset::Constant::create(element::i32, Shape{}, {1});
    const auto zero_boardcast = std::make_shared<default_opset::ConvertLike>(zero_node, x);

    const auto shape = std::make_shared<default_opset::ShapeOf>(x);
    const auto hw = std::make_shared<default_opset::Split>(shape, zero_node, 2);

    // convert to scaler
    const auto h_node = std::make_shared<default_opset::Squeeze>(hw->output(0), zero_node);
    const auto w_node = std::make_shared<default_opset::Squeeze>(hw->output(1), zero_node);

    Output<Node> horizontal_range = std::make_shared<default_opset::Range>(zero_node, w_node, one_node, element::i32);
    Output<Node> vertical_range = std::make_shared<default_opset::Range>(zero_node, h_node, one_node, element::i32);

    if (diagonal != 0) {
        horizontal_range = std::make_shared<default_opset::Subtract>(horizontal_range, diag_node);
    }
    horizontal_range = std::make_shared<default_opset::Unsqueeze>(horizontal_range, zero_node);
    vertical_range = std::make_shared<default_opset::Unsqueeze>(vertical_range, one_node);

    Output<Node> mask;
    if (lower)
        mask = std::make_shared<default_opset::LessEqual>(horizontal_range, vertical_range);
    else
        mask = std::make_shared<default_opset::GreaterEqual>(horizontal_range, vertical_range);
    return node.default_single_output_mapping({std::make_shared<default_opset::Select>(mask, x, zero_boardcast)},
                                              {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov