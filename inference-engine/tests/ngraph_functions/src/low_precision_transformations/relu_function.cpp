// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/relu_function.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ReluFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const std::shared_ptr<Node> relu = std::make_shared<ngraph::opset1::Relu>(dequantizationOp);
    relu->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(relu) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReluFunction");
}

std::shared_ptr<ngraph::Function> ReluFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeFq,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);
    const std::shared_ptr<Node> relu = std::make_shared<ngraph::opset1::Relu>(quantizationOp);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(relu) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReluFunction");
}

std::shared_ptr<ngraph::Function> ReluFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);
    std::shared_ptr<ngraph::opset1::Relu> relu;
    if (quantizationOpBefore->get_output_element_type(0) == precisionAfterOperation) {
        relu = std::make_shared<ngraph::opset1::Relu>(quantizationOpBefore);
    } else {
        relu = std::make_shared<op::TypeRelaxed<ngraph::opset1::Relu>>(quantizationOpBefore);
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(relu, precisionAfterOperation);
    }
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(relu, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ReluFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
