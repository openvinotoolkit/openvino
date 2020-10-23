// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/clamp_function.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ClampFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const std::shared_ptr<Node> clamp = std::make_shared<ngraph::opset1::Clamp>(dequantizationOp, 0, 10);
    clamp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(clamp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampFunction");
}

std::shared_ptr<ngraph::Function> ClampFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    const double clampLowConst,
    const double clampHighConst) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);

    const std::shared_ptr<Node> fq = fakeQuantize.empty() ? nullptr :
        ngraph::builder::makeFakeQuantize(
            input,
            precision,
            fakeQuantize.quantizationLevel,
            fakeQuantize.constantShape,
            fakeQuantize.inputLowValues,
            fakeQuantize.inputHighValues,
            fakeQuantize.outputLowValues,
            fakeQuantize.outputHighValues);

    const std::shared_ptr<ngraph::opset1::Clamp> clamp = std::make_shared<ngraph::opset1::Clamp>(
        fakeQuantize.empty() ? input : fq,
        clampLowConst,
        clampHighConst);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(clamp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampFunction");
}

std::shared_ptr<ngraph::Function> ClampFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    std::shared_ptr<ngraph::opset1::Clamp> clamp = std::make_shared<op::TypeRelaxed<ngraph::opset1::Clamp>>(quantizationOpBefore, 0, 10);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(clamp, precisionAfterOperation);
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(clamp, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
