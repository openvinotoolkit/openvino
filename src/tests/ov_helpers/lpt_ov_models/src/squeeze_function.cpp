// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ov_models/squeeze_function.hpp"

#include "ov_models/subgraph_builders.hpp"
#include "lpt_ov_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> SqueezeFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const auto dequantizationOp = makeDequantization(input, dequantization);

    const auto squeeze = std::make_shared<ov::opset1::Squeeze>(
        dequantizationOp,
        std::make_shared<ov::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));

    squeeze->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(squeeze) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SqueezeTransformation");
}

std::shared_ptr<ov::Model> SqueezeFunction::getOriginal(
    const ov::element::Type originalFunctionPrecision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& axes) {
    const auto input = std::make_shared<ov::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fakeQuantize = fakeQuantizeOnData.empty() ?
        nullptr :
        ov::builder::makeFakeQuantize(
            input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ov::Node> squeeze = std::make_shared<ov::opset1::Squeeze>(
        fakeQuantize == nullptr ? input : fakeQuantize,
        std::make_shared<ov::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(squeeze) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SqueezeTransformation");
}

std::shared_ptr<ov::Model> SqueezeFunction::getReference(
    const ov::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto squeeze = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Squeeze>>(
        op::v0::Squeeze(dequantizationOpBefore, std::make_shared<ov::opset1::Constant>(element::i64, Shape{ axes.size() }, axes)),
        precisionAfterOperation);
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(squeeze, dequantizationAfter);
    dequantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SqueezeTransformation");
}



}  // namespace subgraph
}  // namespace builder
}  // namespace ov
