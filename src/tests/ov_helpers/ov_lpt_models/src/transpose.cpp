// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/transpose.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> TransposeFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<int>& transposeConstValues,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);

    const std::shared_ptr<Node> transpose = std::make_shared<ov::opset1::Transpose>(
        dequantizationOp,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ transposeConstValues.size() }, transposeConstValues));
    transpose->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(transpose) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "TransposeFunction");
}

std::shared_ptr<ov::Model> TransposeFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const std::vector<int>& transposeConstValues,
    const ov::element::Type precisionBeforeFq,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeFq, inputShape);

    const std::shared_ptr<Node> quantizationOp = fqOnData.empty() ?
        std::dynamic_pointer_cast<ov::Node>(input) :
        makeFakeQuantize(input, precisionBeforeFq, fqOnData);

    const std::shared_ptr<Node> transpose = std::make_shared<ov::opset1::Transpose>(
        quantizationOp,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ transposeConstValues.size() }, transposeConstValues));

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(transpose) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "TransposeFunction");
}

std::shared_ptr<ov::Model> TransposeFunction::getReference(
    const ov::PartialShape& inputShape,
    const std::vector<int>& transposeConstValues,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);

    const std::shared_ptr<ov::opset1::Constant> transposeConstant = std::make_shared<ov::opset1::Constant>(
        ov::element::i64,
        ov::Shape{ transposeConstValues.size() },
        transposeConstValues);
    const std::shared_ptr<ov::opset1::Transpose> transpose = std::make_shared<ov::opset1::Transpose>(quantizationOpBefore, transposeConstant);
    if (quantizationOpBefore->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*quantizationOpBefore) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }
    if (transpose->get_output_element_type(0) != precisionAfterOperation) {
        THROW_IE_LPT_EXCEPTION(*transpose) << "unexpected precision '" << precisionAfterOperation << "' after operation";
    }

    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(transpose, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "TransposeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
