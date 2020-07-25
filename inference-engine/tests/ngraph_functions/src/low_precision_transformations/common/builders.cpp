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

    if (!dequantizationOperations.convert.empty()) {
        std::shared_ptr<ngraph::opset1::Convert> convert = std::make_shared<ngraph::opset1::Convert>(
            parent,
            dequantizationOperations.convert.outPrecision);
        parent = convert;
    }

    if (!dequantizationOperations.subtract.empty()) {
        std::shared_ptr<ngraph::opset1::Subtract> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                parent->get_output_element_type(0),
                dequantizationOperations.subtract.values.size() == 1ul ?
                Shape{} :
                Shape{ 1, dequantizationOperations.subtract.values.size(), 1, 1 },
                dequantizationOperations.subtract.values));
        parent = subtract;
    }

    if (!dequantizationOperations.multiply.empty()) {
        std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                parent->get_output_element_type(0),
                dequantizationOperations.multiply.values.size() == 1ul ?
                Shape{} :
                Shape{ 1, dequantizationOperations.multiply.values.size(), 1, 1 },
                dequantizationOperations.multiply.values));
        parent = multiply;
    }

    return parent;
}

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(
    const std::shared_ptr<Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    return as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input,
        precision,
        fqOnData.quantizationLevel,
        fqOnData.constantShape,
        fqOnData.inputLowValues,
        fqOnData.inputHighValues,
        fqOnData.outputLowValues,
        fqOnData.outputHighValues));
}

std::shared_ptr<Node> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<ngraph::opset1::FakeQuantize> fq = makeFakeQuantize(input, precision, fqOnData);
    return std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::FakeQuantize>>(*fq, precision);
}

} // namespace subgraph
} // namespace builder
} // namespace ngraph
