// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/relu.hpp"

#include <memory>

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> ReluFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    const std::shared_ptr<Node> relu = std::make_shared<ov::opset1::Relu>(dequantizationOp);
    relu->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(relu) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ReluFunction");
}

std::shared_ptr<ov::Model> ReluFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);
    const std::shared_ptr<Node> relu = std::make_shared<ov::opset1::Relu>(quantizationOp);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(relu) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ReluFunction");
}

std::shared_ptr<ov::Model> ReluFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);
    std::shared_ptr<ov::opset1::Relu> relu;
    if (quantizationOpBefore->get_output_element_type(0) == precisionAfterOperation) {
        relu = std::make_shared<ov::opset1::Relu>(quantizationOpBefore);
    } else {
        relu = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Relu>>(quantizationOpBefore);
        ov::pass::low_precision::NetworkHelper::setOutDataPrecision(relu, precisionAfterOperation);
    }
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(relu, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ReluFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
