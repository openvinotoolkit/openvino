// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/unsqueeze.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

 std::shared_ptr<ov::Model> UnsqueezeFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const auto dequantizationOp = makeDequantization(input, dequantization);

    const auto unsqueeze = std::make_shared<ov::opset1::Unsqueeze>(
        dequantizationOp,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{axes.size()}, axes));

    unsqueeze->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(unsqueeze) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "UnsqueezeTransformation");
}

std::shared_ptr<ov::Model> UnsqueezeFunction::getOriginal(
    const ov::element::Type originalFunctionPrecision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& axes) {
    const auto input = std::make_shared<ov::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fakeQuantize = fakeQuantizeOnData.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ov::Node> unsqueeze = std::make_shared<ov::opset1::Unsqueeze>(
        fakeQuantize == nullptr ? input : fakeQuantize,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{axes.size()}, axes));

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(unsqueeze) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "UnsqueezeTransformation");
}

std::shared_ptr<ov::Model> UnsqueezeFunction::getReference(
    const ov::PartialShape& inputShape,
    const std::vector<float>& axes,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto unsqueeze = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Unsqueeze>>(
        ov::op::v0::Unsqueeze(dequantizationOpBefore,
                              std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{axes.size()}, axes)),
        precisionAfterOperation);
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(unsqueeze, dequantizationAfter);
    dequantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(dequantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "UnsqueezeTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
