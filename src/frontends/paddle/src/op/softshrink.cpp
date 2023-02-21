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
    auto lambda = node.get_attribute<float>("lambda");

    const auto input_element_type = data.get_element_type();
    std::shared_ptr<default_opset::Constant> negative_lambda =
        default_opset::Constant::create(input_element_type, Shape{}, {-lambda});

    const auto positive_lambda = default_opset::Constant::create(input_element_type, Shape{}, {lambda});

    // Create masks for values below negative lambda and above positive lambda
    std::shared_ptr<ngraph::Node> values_below_neg_lambda =
        std::make_shared<default_opset::Less>(data, negative_lambda);
    std::shared_ptr<ngraph::Node> values_above_pos_lambda =
        std::make_shared<default_opset::Greater>(data, positive_lambda);

    // Convert masks to the same type as input data
    values_below_neg_lambda = std::make_shared<default_opset::Convert>(values_below_neg_lambda, input_element_type);
    values_above_pos_lambda = std::make_shared<default_opset::Convert>(values_above_pos_lambda, input_element_type);

    std::shared_ptr<ngraph::Node> adjusted_inputs =
        default_opset::Constant::create(input_element_type, Shape{data.get_shape()}, {0});
    std::shared_ptr<ngraph::Node> adjusted_inputs_below_neg_lambda =
        std::make_shared<default_opset::Multiply>(std::make_shared<default_opset::Add>(data, positive_lambda),
                                                  values_below_neg_lambda);
    std::shared_ptr<ngraph::Node> adjusted_inputs_above_pos_lambda =
        std::make_shared<default_opset::Multiply>(std::make_shared<default_opset::Add>(data, negative_lambda),
                                                  values_above_pos_lambda);
    adjusted_inputs = std::make_shared<default_opset::Add>(adjusted_inputs, adjusted_inputs_below_neg_lambda);
    adjusted_inputs = std::make_shared<default_opset::Add>(adjusted_inputs, adjusted_inputs_above_pos_lambda);

    return node.default_single_output_mapping({adjusted_inputs}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
