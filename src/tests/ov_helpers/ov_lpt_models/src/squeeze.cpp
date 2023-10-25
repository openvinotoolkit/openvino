// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/squeeze.hpp"

#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> SqueezeFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const auto dequantizationOp = makeDequantization(input, dequantization);

    const auto squeeze = std::make_shared<ngraph::opset1::Squeeze>(
        dequantizationOp,
        std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));

    squeeze->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(squeeze) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SqueezeTransformation");
}

std::shared_ptr<ngraph::Function> SqueezeFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& axes) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fakeQuantize = fakeQuantizeOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ngraph::Node> squeeze = std::make_shared<ngraph::opset1::Squeeze>(
        fakeQuantize == nullptr ? input : fakeQuantize,
        std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(squeeze) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SqueezeTransformation");
}

std::shared_ptr<ngraph::Function> SqueezeFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto squeeze = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Squeeze>>(
        op::Squeeze(dequantizationOpBefore, std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes)),
        precisionAfterOperation);
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(squeeze, dequantizationAfter);
    dequantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SqueezeTransformation");
}



}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
