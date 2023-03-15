// Copyright (C) 2018-2023 Intel Corporation
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
    std::shared_ptr<default_opset::Constant> negative_lambda;
    if (input_element_type.is_signed()) {
        negative_lambda = default_opset::Constant::create(input_element_type, Shape{}, {-lambda});
    } else {
        // Passing -lambd to unsigned type constant will cause an overflow.
        // For unsigned types the lowest possible value is 0.
        negative_lambda = default_opset::Constant::create(input_element_type, Shape{}, {0});
    }
    const auto positive_lambda = default_opset::Constant::create(input_element_type, Shape{}, {lambda});

    // Create masks for values below negative lambda and above positive lambda
    std::shared_ptr<ngraph::Node> values_below_neg_lambda =
        std::make_shared<default_opset::Less>(data, negative_lambda);
    std::shared_ptr<ngraph::Node> values_above_pos_lambda =
        std::make_shared<default_opset::Greater>(data, positive_lambda);

    std::shared_ptr<ngraph::Node> output;
    std::shared_ptr<ngraph::Node> zero_node = default_opset::Constant::create(input_element_type, Shape{}, {0});
    std::shared_ptr<ngraph::Node> positive_node = std::make_shared<default_opset::Add>(data, positive_lambda);
    std::shared_ptr<ngraph::Node> negative_node = std::make_shared<default_opset::Add>(data, negative_lambda);
    output = std::make_shared<default_opset::Select>(values_above_pos_lambda, data, positive_node);
    output = std::make_shared<default_opset::Select>(values_below_neg_lambda, output, negative_node);
    std::shared_ptr<ngraph::Node> zero_mask =
        std::make_shared<default_opset::LogicalOr>(values_below_neg_lambda, values_above_pos_lambda);
    output = std::make_shared<default_opset::Select>(zero_mask, output, zero_node);
    return node.default_single_output_mapping({output}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
