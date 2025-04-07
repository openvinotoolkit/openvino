// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/subtract_multiply_to_multiply_add.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const ov::element::Type precisionAfterDequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    dequantizationOp->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(dequantizationOp) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ov::Model> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precision,
    const ov::builder::subgraph::FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const std::shared_ptr<Node> fq = makeFakeQuantize(input, precision, fqOnData);

    const std::shared_ptr<ov::opset1::Reshape> reshape1 = std::make_shared<ov::opset1::Reshape>(
        fq,
        std::make_shared<ov::opset1::Constant>(
            ov::element::i64,
            Shape({ 3 }),
            std::vector<int64_t>({ inputShape[0].get_length(), inputShape[1].get_length(), -1 })),
        false);

    const std::shared_ptr<ov::opset1::Reshape> reshape2 = std::make_shared<ov::opset1::Reshape>(
        reshape1,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape({ 4 }), inputShape.to_shape()),
        false);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(reshape2) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ov::Model> SubtractMultiplyToMultiplyAddFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const ov::element::Type precisionAfterDequantization,
    const ov::builder::subgraph::Multiply& multiply,
    const ov::builder::subgraph::Add& add) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    std::shared_ptr<Node> parent = dequantizationOp;

    if (!multiply.empty()) {
        parent = makeElementwise<ov::opset1::Multiply>(parent, multiply);
    }

    if (!add.empty()) {
        parent = makeElementwise<ov::opset1::Add>(parent, add);
    }
    parent->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
