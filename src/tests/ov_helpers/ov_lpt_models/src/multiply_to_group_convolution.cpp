// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/multiply_to_group_convolution.hpp"

#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MultiplyToGroupConvolutionFunction::getOriginal(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type& precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const bool haveMultiplyWithNoConstBeforeDequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    std::shared_ptr<ngraph::op::Op> parent = input;
    std::shared_ptr<ngraph::op::Parameter> secondInput;
    if (haveMultiplyWithNoConstBeforeDequantization) {
        secondInput = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);
        parent = std::make_shared<ngraph::opset1::Multiply>(input, secondInput);
    }
    const auto dequantizationOp = makeDequantization(parent, dequantization);
    dequantizationOp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    ngraph::ParameterVector params{input};
    if (haveMultiplyWithNoConstBeforeDequantization) {
        params.push_back(secondInput);
    }
    return std::make_shared<ngraph::Function>(results, params, "MultiplyToGroupConvolutionFunction");
}

std::shared_ptr<ngraph::Function> MultiplyToGroupConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const Constant& constant,
    const bool parentHasOneConsumer) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize = makeFakeQuantize(input, precision, fqOnData);

    const auto rank = inputShape.rank();
    assert(rank.is_static());
    const size_t size = rank.get_length() - 2;
    const auto maxPool = std::make_shared<opset1::MaxPool>(
        fakeQuantize,
        Strides(size, 1),
        Shape(size, 1),
        Shape(size, 0),
        Shape(size, 2));

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(
        maxPool,
        std::make_shared<ngraph::opset1::Constant>(constant.outPrecision, constant.shape, constant.values));
    multiply->set_friendly_name("output");

    ngraph::ResultVector results = parentHasOneConsumer ?
        ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(multiply)} :
        ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(maxPool), std::make_shared<ngraph::opset1::Result>(multiply)};
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MultiplyToGroupConvolutionFunction");
}

std::shared_ptr<ngraph::Function> MultiplyToGroupConvolutionFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type& inputPrecision,
    const std::shared_ptr<ngraph::opset1::Constant>& weights,
    const std::shared_ptr<ngraph::opset1::Constant>& biases,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);

    const size_t spatialDimsSize = inputShape.rank().get_length() - 2;
    ngraph::Strides strides(spatialDimsSize, 1ul);
    ngraph::CoordinateDiff pads(spatialDimsSize, 0ul);
    ngraph::Strides dilations(spatialDimsSize, 1ul);

    const auto gconv = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{ element::f32 },
        ov::op::TemporaryReplaceOutputType(input, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weights, element::f32).get(),
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
