// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/mvn_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MVNFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const AxisSet& reductionAxes,
    const bool& normalizeVariance,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const auto dequantizationOp = makeDequantization(input, dequantization);
    const auto mvn = std::make_shared<ngraph::op::MVN>(dequantizationOp, reductionAxes, normalizeVariance);
    mvn->set_friendly_name("output");
    auto& rtInfo = mvn->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("mvn");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(mvn) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MVNFunction");
}

std::shared_ptr<ngraph::Function> MVNFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const AxisSet& reductionAxes,
    const bool& normalizeVariance) {
    float k = 50.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = ngraph::builder::makeFakeQuantize(
        input, precision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    const auto mvn = std::make_shared<ngraph::op::MVN>(fakeQuantizeOnActivations, reductionAxes, normalizeVariance);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(mvn) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MVNFunction");
}

std::shared_ptr<ngraph::Function> MVNFunction::getReference(
    const ngraph::Shape& inputShape,
    const AxisSet& reductionAxes,
    const bool& normalizeVariance,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto mvn = std::make_shared<ngraph::op::TypeRelaxed<ngraph::op::MVN>>(
        op::MVN(dequantizationOpBefore, reductionAxes, normalizeVariance), precisionAfterOperation);
    auto& rtInfo = mvn->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("mvn");

    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(mvn, dequantizationAfter);
    dequantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MVNFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
