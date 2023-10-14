// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/subtract_multiply_to_multiply_add.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ov_lpt_models/common/builders.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    dequantizationOp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precision,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const std::shared_ptr<Node> fq = makeFakeQuantize(input, precision, fqOnData);

    const std::shared_ptr<ngraph::opset1::Reshape> reshape1 = std::make_shared<ngraph::opset1::Reshape>(
        fq,
        std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i64,
            Shape({ 3 }),
            std::vector<int64_t>({ inputShape[0].get_length(), inputShape[1].get_length(), -1 })),
        false);

    const std::shared_ptr<ngraph::opset1::Reshape> reshape2 = std::make_shared<ngraph::opset1::Reshape>(
        reshape1,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, Shape({ 4 }), inputShape.to_shape()),
        false);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::builder::subgraph::Multiply& multiply,
    const ngraph::builder::subgraph::Add& add) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    std::shared_ptr<Node> parent = dequantizationOp;

    if (!multiply.empty()) {
        parent = makeElementwise<opset1::Multiply>(parent, multiply);
    }

    if (!add.empty()) {
        parent = makeElementwise<opset1::Add>(parent, add);
    }
    parent->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
