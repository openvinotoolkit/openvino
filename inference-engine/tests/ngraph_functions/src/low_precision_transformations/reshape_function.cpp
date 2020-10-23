// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/reshape_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ReshapeFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);

    const std::shared_ptr<Node> reshape = std::make_shared<ngraph::opset1::Reshape>(
        dequantizationOp,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ reshapeConstValues.size() }, reshapeConstValues),
        true);
    reshape->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReshapeFunction");
}

std::shared_ptr<ngraph::Function> ReshapeFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ngraph::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeFq,
        ngraph::Shape(inputShape));

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
    const ngraph::Shape& inputShape,
    const std::vector<int>& reshapeConstValues,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    const std::shared_ptr<ngraph::opset1::Constant> reshapeConstant = std::make_shared<ngraph::opset1::Constant>(
        ngraph::element::i64,
        ngraph::Shape{ reshapeConstValues.size() },
        reshapeConstValues);
    const std::shared_ptr<ngraph::opset1::Reshape> reshape = std::make_shared<ngraph::opset1::Reshape>(quantizationOpBefore, reshapeConstant, true);
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
