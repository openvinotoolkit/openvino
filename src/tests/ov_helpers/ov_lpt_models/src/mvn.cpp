// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/mvn.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MVNFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::AxisSet& reductionAxes,
    const bool& normalizeVariance,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const int opset_version) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    auto deqStructure = dequantization;
    deqStructure.multiply.outPrecision = precision;
    const auto dequantizationOp = makeDequantization(input, deqStructure);
    std::shared_ptr<Node> mvn;
    if (opset_version == 2) {
        mvn = std::make_shared<ov::op::v0::MVN>(dequantizationOp, reductionAxes, normalizeVariance);
    } else if (opset_version == 6) {
        mvn = std::make_shared<ov::op::v6::MVN>(dequantizationOp,
                                                std::make_shared<ov::opset1::Constant>(ov::element::i64,
                                                                                       Shape{reductionAxes.size()},
                                                                                       reductionAxes.to_vector()),
                                                normalizeVariance,
                                                1e-9,
                                                ov::op::MVNEpsMode::INSIDE_SQRT);
    }
    mvn->set_friendly_name("output");
    auto& rtInfo = mvn->get_rt_info();
    rtInfo["Variant::std::string"] = "mvn";

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(mvn) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MVNFunction");
}

std::shared_ptr<ov::Model> MVNFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::AxisSet& reductionAxes,
    const bool& normalizeVariance) {
    float k = 50.f;

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = ov::test::utils::make_fake_quantize(
        input, precision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    const auto mvn = std::make_shared<ov::op::v0::MVN>(fakeQuantizeOnActivations, reductionAxes, normalizeVariance);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(mvn) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MVNFunction");
}

std::shared_ptr<ov::Model> MVNFunction::getReference(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::AxisSet& reductionAxes,
    const bool& normalizeVariance,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int opset_version) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    auto deqBeforeStructure = dequantizationBefore;
    deqBeforeStructure.multiply.outPrecision = precision;
    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, deqBeforeStructure);
    std::shared_ptr<Node> mvn;
    if (opset_version == 2) {
        mvn = std::make_shared<ov::op::TypeRelaxed<ov::op::v0::MVN>>(
            ov::op::v0::MVN(dequantizationOpBefore, reductionAxes, normalizeVariance),
            dequantizationAfter.empty() ? precision : ov::element::f32);
    } else if (opset_version == 6) {
        mvn = std::make_shared<ov::op::TypeRelaxed<ov::op::v6::MVN>>(
            ov::op::v6::MVN(dequantizationOpBefore,
                            std::make_shared<ov::opset1::Constant>(ov::element::i64,
                                                                   Shape{reductionAxes.size()},
                                                                   reductionAxes.to_vector()),
                            normalizeVariance,
                            1e-9,
                            ov::op::MVNEpsMode::INSIDE_SQRT),
            dequantizationAfter.empty() ? precision : ov::element::f32);
    }
    auto& rtInfo = mvn->get_rt_info();
    rtInfo["Variant::std::string"] = "mvn";

    auto deqAfterStructure = dequantizationAfter;
    deqAfterStructure.multiply.outPrecision = precision;
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(mvn, deqAfterStructure);
    dequantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MVNFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
