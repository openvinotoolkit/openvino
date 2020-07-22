// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

#include <queue>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<Node> makeDequantization(
    const std::shared_ptr<ngraph::Node> data,
    const DequantizationOperations& dequantizationOperations) {
    std::shared_ptr<ngraph::Node> parent = data;

    // TODO: FIXME: dequantizationOperations.convert.empty()
    if (dequantizationOperations.convert.outPrecision != ngraph::element::undefined) {
        std::shared_ptr<ngraph::opset1::Convert> convert = std::make_shared<ngraph::opset1::Convert>(
            parent,
            dequantizationOperations.convert.outPrecision);
        parent = convert;
    }

    if (!dequantizationOperations.subtractValues.empty()) {
        std::shared_ptr<ngraph::opset1::Subtract> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                parent->get_output_element_type(0),
                dequantizationOperations.subtractValues.size() == 1ul ?
                Shape{} :
                Shape{ 1, dequantizationOperations.subtractValues.size(), 1, 1 },
                dequantizationOperations.subtractValues));
        parent = subtract;
    }

    if (!dequantizationOperations.multiplyValues.empty()) {
        std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                parent->get_output_element_type(0),
                dequantizationOperations.multiplyValues.size() == 1ul ?
                Shape{} :
                Shape{ 1, dequantizationOperations.multiplyValues.size(), 1, 1 },
                dequantizationOperations.multiplyValues));
        parent = multiply;
    }

    return parent;
}

std::shared_ptr<Node> makeFakeQuantize(
    const std::shared_ptr<Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    return ngraph::builder::makeFakeQuantize(
        input,
        precision,
        fqOnData.quantizationLevel,
        fqOnData.constantShape,
        fqOnData.inputLowValues,
        fqOnData.inputHighValues,
        fqOnData.outputLowValues,
        fqOnData.outputHighValues);
}

std::shared_ptr<Node> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    return ngraph::builder::makeFakeQuantizeTypeRelaxed(
        input,
        precision,
        fqOnData.quantizationLevel,
        fqOnData.constantShape,
        fqOnData.inputLowValues,
        fqOnData.inputHighValues,
        fqOnData.outputLowValues,
        fqOnData.outputHighValues);
}

} // namespace subgraph
} // namespace builder
} // namespace ngraph
