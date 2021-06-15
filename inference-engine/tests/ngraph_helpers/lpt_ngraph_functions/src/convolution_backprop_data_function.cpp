// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/convolution_backprop_data_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "low_precision/common/dequantization_op.hpp"
#include "low_precision/network_helper.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<Function> ConvolutionBackpropDataFunction::get(
    const element::Type netPrecision,
    const Shape& inputShape,
    const Shape& outputShape,
    const builder::subgraph::FakeQuantizeOnData& fqOnData,
    const std::shared_ptr<Node>& weights) {
    const auto input = std::make_shared<opset1::Parameter>(netPrecision, inputShape);
    const auto fq = makeFakeQuantize(input, netPrecision, fqOnData);

    auto convolutionBackpropData = std::make_shared<opset1::ConvolutionBackpropData>(
        fq,
        weights,
        Strides{ 1, 1 },
        CoordinateDiff{ 0, 0 },
        CoordinateDiff{ 0, 0 },
        Strides{ 1, 1 });
    convolutionBackpropData->set_friendly_name("convolutionBackpropData");

    ngraph::ResultVector results{ std::make_shared<opset1::Result>(convolutionBackpropData) };
    return std::make_shared<ngraph::Function>(results, ParameterVector{ input }, "ConvolutionBackpropDataTransformation");
}

std::shared_ptr<Node> ConvolutionBackpropDataFunction::getWeights(
    const Shape& shape,
    const element::Type& netPrecision,
    const builder::subgraph::FakeQuantizeOnWeights& fqOnWeights,
    const std::shared_ptr<opset1::Constant>& value) {
    const auto weights = value != nullptr ?
            value :
            std::make_shared<opset1::Constant>(
            element::i8,
            shape,
            std::vector<float>(shape_size(shape), 1));
    const auto convert = std::make_shared<opset1::Convert>(weights, netPrecision);
    OutputVector convertedOutput(1);
    convert->constant_fold(convertedOutput, convert->input_values());
    const auto convertedWeights = convertedOutput[0].get_node_shared_ptr();
    const auto fq = makeFakeQuantize(convertedWeights, netPrecision, fqOnWeights);

    return fq;
}

std::shared_ptr<Node> ConvolutionBackpropDataFunction::getWeights(
    const Shape& shape,
    const element::Type& netPrecision,
    const builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const std::shared_ptr<opset1::Constant>& value) {
    auto weights =
        value != nullptr ?
            value :
            std::make_shared<opset1::Constant>(
                element::i8,
                shape,
                std::vector<float>(shape_size(shape), 1));
    auto dequantizationStructure = dequantizationOnWeights;
    dequantizationStructure.setPrecision(netPrecision);
    if (!dequantizationOnWeights.subtract.constantPrecision.is_real()) {
        dequantizationStructure.subtract.constantPrecision = dequantizationOnWeights.subtract.constantPrecision;
    }
    if (weights->get_element_type().is_real()) {
        weights = as_type_ptr<opset1::Constant>(fold<opset1::Convert>(weights, netPrecision));
    }
    const auto dq = makeDequantization(weights, dequantizationStructure);

    return dq;
}

std::shared_ptr<Function> ConvolutionBackpropDataFunction::getOriginal(
    const element::Type precision,
    const element::Type netPrecision,
    const Shape& inputShape,
    const Shape& outputShape,
    const builder::subgraph::DequantizationOperations& dequantization,
    const std::shared_ptr<Node>& weights) {
    const auto input = std::make_shared<opset1::Parameter>(precision, inputShape);
    auto dequantizationStructure = dequantization;
    dequantizationStructure.multiply.outPrecision = netPrecision;
    const auto activations = makeDequantization(input, dequantizationStructure);

    auto convolutionBackpropData = std::make_shared<opset1::ConvolutionBackpropData>(
            activations,
            weights,
            Strides{ 1, 1 },
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            Strides{ 1, 1 });

    convolutionBackpropData->set_friendly_name("output");
    ngraph::ResultVector results{ std::make_shared<opset1::Result>(convolutionBackpropData) };
    return std::make_shared<ngraph::Function>(results, ParameterVector{ input }, "ConvolutionBackpropDataTransformation");
}

std::shared_ptr<Function>  ConvolutionBackpropDataFunction::getReference(
    const element::Type precision,
    const element::Type netPrecision,
    const Shape& inputShape,
    const Shape& outputShape,
    const builder::subgraph::DequantizationOperations& dequantization,
    const std::shared_ptr<Node>& weights,
    const builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<opset1::Parameter>(precision, inputShape);
    auto dequantizationStructure = dequantization;
    dequantizationStructure.multiply.outPrecision = netPrecision;
    const auto activations = makeDequantization(input, dequantizationStructure);

    auto convolutionBackpropData = std::make_shared<op::TypeRelaxed<opset1::ConvolutionBackpropData>>(
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{ dequantizationAfter.empty() ? netPrecision : element::f32 },
            ngraph::op::TemporaryReplaceOutputType(activations, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(weights, element::f32).get(),
            Strides{ 1, 1 },
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            Strides{ 1, 1 });

    auto dequantizationStructureAfter = dequantizationAfter;
    dequantizationStructureAfter.multiply.outPrecision = netPrecision;
    const auto result = makeDequantization(convolutionBackpropData, dequantizationStructureAfter);
    result->set_friendly_name("output");
    ngraph::ResultVector results{ std::make_shared<opset1::Result>(result) };
    return std::make_shared<ngraph::Function>(results, ParameterVector{ input }, "ConvolutionBackpropDataTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
