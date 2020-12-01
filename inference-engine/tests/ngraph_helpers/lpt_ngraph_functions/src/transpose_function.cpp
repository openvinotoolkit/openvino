// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/transpose_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> TransposeFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const std::vector<int>& transposeConstValues,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);

    const std::shared_ptr<Node> transpose = std::make_shared<ngraph::opset1::Transpose>(
        dequantizationOp,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ transposeConstValues.size() }, transposeConstValues));
    transpose->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(transpose) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "TransposeFunction");
}

std::shared_ptr<ngraph::Function> TransposeFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const std::vector<int>& transposeConstValues,
    const ngraph::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeFq,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);

    const std::shared_ptr<Node> transpose = std::make_shared<ngraph::opset1::Transpose>(
        quantizationOp,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{ transposeConstValues.size() }, transposeConstValues));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(transpose) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "TransposeFunction");
}

std::shared_ptr<ngraph::Function> TransposeFunction::getReference(
    const ngraph::Shape& inputShape,
    const std::vector<int>& transposeConstValues,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    const std::shared_ptr<ngraph::opset1::Constant> transposeConstant = std::make_shared<ngraph::opset1::Constant>(
        ngraph::element::i64,
        ngraph::Shape{ transposeConstValues.size() },
        transposeConstValues);
    const std::shared_ptr<ngraph::opset1::Transpose> transpose = std::make_shared<ngraph::opset1::Transpose>(quantizationOpBefore, transposeConstant);
    if (quantizationOpBefore->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*quantizationOpBefore) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }
    if (transpose->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*transpose) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }

    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(transpose, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "TransposeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
