// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/reshape_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

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
        as_type_ptr<ngraph::Node>(input) :
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
    std::shared_ptr<ngraph::opset1::Reshape> reshape;
    std::shared_ptr<ngraph::opset1::Constant> reshapeConstant = std::make_shared<ngraph::opset1::Constant>(
        ngraph::element::i64,
        ngraph::Shape{ reshapeConstValues.size() },
        reshapeConstValues);
    if (quantizationOpBefore->get_output_element_type(0) == precisionAfterOperation) {
        reshape = std::make_shared<ngraph::opset1::Reshape>(quantizationOpBefore, reshapeConstant, true);
    } else {
        reshape = std::make_shared<op::TypeRelaxed<ngraph::opset1::Reshape>>(quantizationOpBefore, reshapeConstant, true);
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(reshape, precisionAfterOperation);
    }
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(reshape, dequantizationAfter);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReshapeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
