// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/prelu.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "ov_ops/type_relaxed.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> PReluFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const auto slope = std::make_shared<ngraph::opset1::Constant>(precisionBeforeDequantization, Shape{}, std::vector<float> { 0.1f });
    const auto prelu = std::make_shared<ngraph::opset1::PRelu>(dequantizationOp, slope);
    prelu->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(prelu) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "PReluFunction");
}

std::shared_ptr<ngraph::Function> PReluFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);
    const auto slope = std::make_shared<ngraph::opset1::Constant>(precisionBeforeFq, Shape{}, std::vector<float> { 0.1f });
    const auto prelu = std::make_shared<ngraph::opset1::PRelu>(quantizationOp, slope);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(prelu) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "PReluFunction");
}

std::shared_ptr<ngraph::Function> PReluFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto slope = std::make_shared<ngraph::opset1::Constant>(precisionBeforeDequantization, Shape{}, std::vector<float> { 0.1f });
    const auto prelu = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::PRelu>>(
        ngraph::opset1::PRelu(quantizationOpBefore, slope),
        precisionAfterOperation);
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(prelu, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "PReluFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
