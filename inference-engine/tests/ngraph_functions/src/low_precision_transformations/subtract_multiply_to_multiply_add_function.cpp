// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/subtract_multiply_to_multiply_add_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::builder::subgraph::Multiply& multiply,
    const ngraph::builder::subgraph::Add& add) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    std::shared_ptr<Node> parent = dequantizationOp;

    if (!multiply.empty()) {
        parent = makeElementwise<ngraph::opset1::Multiply>(parent, multiply);
    }

    if (!add.empty()) {
        parent = makeElementwise<ngraph::opset1::Add>(parent, add);
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
