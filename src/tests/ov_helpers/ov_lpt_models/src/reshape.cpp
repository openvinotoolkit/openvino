// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/reshape.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ReshapeFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);

    std::shared_ptr<Node> reshape_pattern;
    if (!reshapeConstValues.empty()) {
        reshape_pattern = opset1::Constant::create(element::i64, Shape{ reshapeConstValues.size() }, reshapeConstValues);
    } else {
        reshape_pattern = std::make_shared<opset1::ShapeOf>(dequantizationOp);
    }

    const auto reshape = std::make_shared<ngraph::opset1::Reshape>(dequantizationOp, reshape_pattern, true);
    reshape->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReshapeFunction");
}

std::shared_ptr<ngraph::Function> ReshapeFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ngraph::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);

    const std::shared_ptr<Node> reshape = std::make_shared<ngraph::opset1::Reshape>(
        quantizationOp,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ reshapeConstValues.size() }, reshapeConstValues),
        true);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReshapeFunction");
}

std::shared_ptr<ngraph::Function> ReshapeFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    std::shared_ptr<Node> reshape_pattern;
    if (!reshapeConstValues.empty()) {
        reshape_pattern = opset1::Constant::create(element::i64, Shape{ reshapeConstValues.size() }, reshapeConstValues);
    } else {
        reshape_pattern = makeDequantization(quantizationOpBefore, dequantizationAfter);
        reshape_pattern = std::make_shared<opset1::ShapeOf>(reshape_pattern);
    }

    const auto reshape = std::make_shared<opset1::Reshape>(quantizationOpBefore, reshape_pattern, true);
    if (quantizationOpBefore->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*quantizationOpBefore) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }
    if (reshape->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*reshape) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }

    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(reshape, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReshapeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
