// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs softshrink(const NodeContext& node) {
    auto data = node.get_input("X");
    const float lambda = node.get_attribute<float>("lambda", 0.5f);
    const auto input_element_type = data.get_element_type();
    PADDLE_OP_CHECK(node, lambda >= 0, "Softshrink op lambda must be non-negative.");
    PADDLE_OP_CHECK(node, input_element_type.is_signed(), "Softshrink op input must be signed type.");

    std::shared_ptr<Node> output;
    const auto positive_lambda = default_opset::Constant::create(input_element_type, Shape{}, {lambda});
    const auto negative_lambda = default_opset::Constant::create(input_element_type, Shape{}, {-lambda});
    std::shared_ptr<Node> negative_node = std::make_shared<default_opset::Subtract>(data, positive_lambda);
    std::shared_ptr<Node> positive_node = std::make_shared<default_opset::Add>(data, positive_lambda);

    std::shared_ptr<Node> zero_node = default_opset::Constant::create(input_element_type, Shape{}, {0});

    // Create masks for values below negative lambda and above positive lambda
    std::shared_ptr<Node> values_below_neg_lambda = std::make_shared<default_opset::Less>(data, negative_lambda);
    std::shared_ptr<Node> values_above_pos_lambda = std::make_shared<default_opset::Greater>(data, positive_lambda);

    output = std::make_shared<default_opset::Select>(values_above_pos_lambda, negative_node, data);
    output = std::make_shared<default_opset::Select>(values_below_neg_lambda, positive_node, output);

    std::shared_ptr<Node> zero_mask =
        std::make_shared<default_opset::LogicalOr>(values_below_neg_lambda, values_above_pos_lambda);

    output = std::make_shared<default_opset::Select>(zero_mask, output, zero_node);

    return node.default_single_output_mapping({output}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
