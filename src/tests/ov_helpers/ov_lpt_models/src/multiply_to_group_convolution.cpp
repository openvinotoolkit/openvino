// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/multiply_to_group_convolution.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MultiplyToGroupConvolutionFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type& precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization,
    const bool haveMultiplyWithNoConstBeforeDequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    std::shared_ptr<ov::op::Op> parent = input;
    std::shared_ptr<ov::op::v0::Parameter> secondInput;
    if (haveMultiplyWithNoConstBeforeDequantization) {
        secondInput = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
        parent = std::make_shared<ov::opset1::Multiply>(input, secondInput);
    }
    const auto dequantizationOp = makeDequantization(parent, dequantization);
    dequantizationOp->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(dequantizationOp) };
    ov::ParameterVector params{input};
    if (haveMultiplyWithNoConstBeforeDequantization) {
        params.push_back(secondInput);
    }
    return std::make_shared<ov::Model>(results, params, "MultiplyToGroupConvolutionFunction");
}

std::shared_ptr<ov::Model> MultiplyToGroupConvolutionFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const Constant& constant,
    const bool parentHasOneConsumer) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize = makeFakeQuantize(input, precision, fqOnData);

    const auto rank = inputShape.rank();
    assert(rank.is_static());
    const size_t size = rank.get_length() - 2;
    const auto maxPool = std::make_shared<ov::opset1::MaxPool>(
        fakeQuantize,
        Strides(size, 1),
        Shape(size, 1),
        Shape(size, 0),
        Shape(size, 2));

    const auto multiply = std::make_shared<ov::opset1::Multiply>(
        maxPool,
        std::make_shared<ov::opset1::Constant>(constant.outPrecision, constant.shape, constant.values));
    multiply->set_friendly_name("output");

    ov::ResultVector results = parentHasOneConsumer ?
        ov::ResultVector{std::make_shared<ov::opset1::Result>(multiply)} :
        ov::ResultVector{std::make_shared<ov::opset1::Result>(maxPool), std::make_shared<ov::opset1::Result>(multiply)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MultiplyToGroupConvolutionFunction");
}

std::shared_ptr<ov::Model> MultiplyToGroupConvolutionFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::element::Type& inputPrecision,
    const std::shared_ptr<ov::opset1::Constant>& weights,
    const std::shared_ptr<ov::opset1::Constant>& biases,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);

    const size_t spatialDimsSize = inputShape.rank().get_length() - 2;
    ov::Strides strides(spatialDimsSize, 1ul);
    ov::CoordinateDiff pads(spatialDimsSize, 0ul);
    ov::Strides dilations(spatialDimsSize, 1ul);

    const auto gconv = std::make_shared<ov::op::TypeRelaxed<ov::opset1::GroupConvolution>>(
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{ov::element::f32},
        ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
        strides,
        pads,
        pads,
        dilations);
    std::shared_ptr<ov::Node> lastNode = gconv;
    if (biases) {
        lastNode = std::make_shared<ov::opset1::Add>(gconv, biases);
    }
    const auto dequantizationOp = makeDequantization(lastNode, dequantization);
    dequantizationOp->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(dequantizationOp) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MultiplyToGroupConvolutionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
