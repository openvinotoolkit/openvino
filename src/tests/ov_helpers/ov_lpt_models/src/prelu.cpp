// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/prelu.hpp"

#include <memory>

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> PReluFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const auto slope = std::make_shared<ov::opset1::Constant>(precisionBeforeDequantization, Shape{}, std::vector<float> { 0.1f });
    const auto prelu = std::make_shared<ov::opset1::PRelu>(dequantizationOp, slope);
    prelu->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(prelu) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "PReluFunction");
}

std::shared_ptr<ov::Model> PReluFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);
    const auto slope = std::make_shared<ov::opset1::Constant>(precisionBeforeFq, Shape{}, std::vector<float> { 0.1f });
    const auto prelu = std::make_shared<ov::opset1::PRelu>(quantizationOp, slope);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(prelu) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "PReluFunction");
}

std::shared_ptr<ov::Model> PReluFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto slope = std::make_shared<ov::opset1::Constant>(precisionBeforeDequantization, Shape{}, std::vector<float> { 0.1f });
    const auto prelu = std::make_shared<ov::op::TypeRelaxed<ov::opset1::PRelu>>(
        ov::opset1::PRelu(quantizationOpBefore, slope),
        precisionAfterOperation);
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(prelu, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "PReluFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
