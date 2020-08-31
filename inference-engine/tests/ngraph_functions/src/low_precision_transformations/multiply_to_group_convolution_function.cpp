// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/multiply_to_group_convolution_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MultiplyToGroupConvolutionFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type& precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const auto dequantizationOp = makeDequantization(input, dequantization);
    dequantizationOp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MultiplyToGroupConvolutionFunction");
}

std::shared_ptr<ngraph::Function> MultiplyToGroupConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = makeFakeQuantize(input, precision, fqOnData);
    const auto reshape = std::make_shared<ngraph::opset1::Reshape>(
        fakeQuantizeOnActivations,
        std::make_shared<ngraph::opset1::Constant>(element::i32, Shape{ inputShape.size() }, inputShape),
        true);
    reshape->set_friendly_name("output");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(reshape)
    };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MultiplyToGroupConvolutionFunction");
}

std::shared_ptr<ngraph::Function> MultiplyToGroupConvolutionFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type& inputPrecision,
    const std::shared_ptr<ngraph::opset1::Constant>& weights,
    const std::shared_ptr<ngraph::opset1::Constant>& biases,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        inputPrecision,
        ngraph::Shape(inputShape));

    const size_t spatialDimsSize = inputShape.size() - 2;
    ngraph::Strides strides(spatialDimsSize, 1ul);
    ngraph::CoordinateDiff pads(spatialDimsSize, 0ul);
    ngraph::Strides dilations(spatialDimsSize, 1ul);

    const auto gconv = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{ element::f32 },
        ngraph::op::TemporaryReplaceOutputType(input, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(weights, element::f32).get(),
        strides,
        pads,
        pads,
        dilations);
    std::shared_ptr<ngraph::Node> lastNode = gconv;
    if (biases) {
        lastNode = std::make_shared<ngraph::opset1::Add>(gconv, biases);
    }
    const auto dequantizationOp = makeDequantization(lastNode, dequantization);
    dequantizationOp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MultiplyToGroupConvolutionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
