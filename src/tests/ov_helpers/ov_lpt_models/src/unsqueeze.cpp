// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/unsqueeze.hpp"

#include "ov_models/builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

 std::shared_ptr<ngraph::Function> UnsqueezeFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const auto dequantizationOp = makeDequantization(input, dequantization);

    const auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
        dequantizationOp,
        std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));

    unsqueeze->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(unsqueeze) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "UnsqueezeTransformation");
}

std::shared_ptr<ngraph::Function> UnsqueezeFunction::getOriginal(
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

    const std::shared_ptr<ngraph::Node> unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
        fakeQuantize == nullptr ? input : fakeQuantize,
        std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(unsqueeze) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "UnsqueezeTransformation");
}

std::shared_ptr<ngraph::Function> UnsqueezeFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto unsqueeze = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Unsqueeze>>(
        op::v0::Unsqueeze(dequantizationOpBefore, std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes)),
        precisionAfterOperation);
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(unsqueeze, dequantizationAfter);
    dequantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "UnsqueezeTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
