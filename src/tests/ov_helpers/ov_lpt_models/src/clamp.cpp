// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>


#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/clamp.hpp"
#include "low_precision/network_helper.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> ClampFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const std::shared_ptr<Node> clamp = std::make_shared<ov::opset1::Clamp>(dequantizationOp, 0, 10);
    clamp->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(clamp) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ClampFunction");
}

std::shared_ptr<ov::Model> ClampFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    const double clampLowConst,
    const double clampHighConst) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);

    const std::shared_ptr<Node> fq = fakeQuantize.empty() ? nullptr :
        ov::test::utils::make_fake_quantize(
            input,
            precision,
            fakeQuantize.quantizationLevel,
            fakeQuantize.constantShape,
            fakeQuantize.inputLowValues,
            fakeQuantize.inputHighValues,
            fakeQuantize.outputLowValues,
            fakeQuantize.outputHighValues);

    const std::shared_ptr<ov::opset1::Clamp> clamp = std::make_shared<ov::opset1::Clamp>(
        fakeQuantize.empty() ? input : fq,
        clampLowConst,
        clampHighConst);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(clamp) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ClampFunction");
}

std::shared_ptr<ov::Model> ClampFunction::getWithNonDequantizationMultiply(
    const ov::PartialShape& inputShape,
    const ov::element::Type precision) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);

    const auto multiply = std::make_shared<ov::opset1::Multiply>(input1, input2);
    const auto clamp = std::make_shared<ov::opset1::Clamp>(multiply, 0.0, 6.0);
    clamp->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(clamp) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input1, input2 }, "ClampFunction");
}

std::shared_ptr<ov::Model> ClampFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    std::shared_ptr<ov::opset1::Clamp> clamp = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Clamp>>(quantizationOpBefore, 0, 10);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(clamp, precisionAfterOperation);
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(clamp, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ClampFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
