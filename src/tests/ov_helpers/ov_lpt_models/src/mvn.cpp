// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/mvn.hpp"

#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MVNFunction::getOriginal(
    const element::Type precision,
    const ngraph::PartialShape& inputShape,
    const AxisSet& reductionAxes,
    const bool& normalizeVariance,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const int opset_version) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    auto deqStructure = dequantization;
    deqStructure.multiply.outPrecision = precision;
    const auto dequantizationOp = makeDequantization(input, deqStructure);
    std::shared_ptr<Node> mvn;
    if (opset_version == 2) {
        mvn = std::make_shared<ngraph::op::MVN>(dequantizationOp, reductionAxes, normalizeVariance);
    } else if (opset_version == 6) {
        mvn = std::make_shared<ngraph::opset6::MVN>(
                dequantizationOp,
                std::make_shared<opset1::Constant>(element::i64, Shape{reductionAxes.size()}, reductionAxes.to_vector()),
                normalizeVariance,
                1e-9,
                op::MVNEpsMode::INSIDE_SQRT);
    }
    mvn->set_friendly_name("output");
    auto& rtInfo = mvn->get_rt_info();
    rtInfo["Variant::std::string"] = "mvn";

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(mvn) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MVNFunction");
}

std::shared_ptr<ngraph::Function> MVNFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
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
    const element::Type precision,
    const ngraph::PartialShape& inputShape,
    const AxisSet& reductionAxes,
    const bool& normalizeVariance,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int opset_version) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    auto deqBeforeStructure = dequantizationBefore;
    deqBeforeStructure.multiply.outPrecision = precision;
    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, deqBeforeStructure);
    std::shared_ptr<Node> mvn;
    if (opset_version == 2) {
        mvn = std::make_shared<ov::op::TypeRelaxed<ngraph::op::MVN>>(
            op::MVN(dequantizationOpBefore, reductionAxes, normalizeVariance),
            dequantizationAfter.empty() ? precision : element::f32);
    } else if (opset_version == 6) {
        mvn = std::make_shared<ov::op::TypeRelaxed<ngraph::opset6::MVN>>(
            opset6::MVN(dequantizationOpBefore,
                std::make_shared<opset1::Constant>(element::i64, Shape{reductionAxes.size()}, reductionAxes.to_vector()),
                normalizeVariance,
                1e-9,
                op::MVNEpsMode::INSIDE_SQRT),
            dequantizationAfter.empty() ? precision : element::f32);
    }
    auto& rtInfo = mvn->get_rt_info();
    rtInfo["Variant::std::string"] = "mvn";

    auto deqAfterStructure = dequantizationAfter;
    deqAfterStructure.multiply.outPrecision = precision;
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(mvn, deqAfterStructure);
    dequantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MVNFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
