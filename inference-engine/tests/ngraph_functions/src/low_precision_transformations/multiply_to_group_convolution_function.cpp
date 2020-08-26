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
    const AxisSet& reductionAxes,
    const bool& normalizeVariance) {
    float k = 50.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = ngraph::builder::makeFakeQuantize(
        input, precision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
    const auto mvn = std::make_shared<ngraph::op::MVN>(fakeQuantizeOnActivations, reductionAxes, normalizeVariance);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(mvn) };
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

    const auto gconv = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
        ngraph::opset1::GroupConvolution(
            ngraph::op::TemporaryReplaceOutputType(input, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(weights, element::f32).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 }),
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});
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
