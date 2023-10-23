// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/clamp.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ClampFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const std::shared_ptr<Node> clamp = std::make_shared<ngraph::opset1::Clamp>(dequantizationOp, 0, 10);
    clamp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(clamp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampFunction");
}

std::shared_ptr<ngraph::Function> ClampFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
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

std::shared_ptr<ngraph::Function> ClampFunction::getWithNonDequantizationMultiply(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precision) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(input1, input2);
    const auto clamp = std::make_shared<ngraph::opset1::Clamp>(multiply, 0.0, 6.0);
    clamp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(clamp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1, input2 }, "ClampFunction");
}

std::shared_ptr<ngraph::Function> ClampFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    std::shared_ptr<ngraph::opset1::Clamp> clamp = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Clamp>>(quantizationOpBefore, 0, 10);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(clamp, precisionAfterOperation);
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(clamp, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
