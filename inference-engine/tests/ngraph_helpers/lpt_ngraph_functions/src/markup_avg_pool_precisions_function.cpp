// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "low_precision/network_helper.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

#include "lpt_ngraph_functions/markup_avg_pool_precisions_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MarkupAvgPoolPrecisionsFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    auto deqBeforeStructure = dequantizationBefore;
    deqBeforeStructure.multiply.outPrecision = precision;
    // const auto parent = makeDequantization(input, deqBeforeStructure);

    parent = ngraph::builder::makeFakeQuantize(input, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
    parent->set_friendly_name("fakeQuantizeOnActivations");

    parent = std::make_shared<ngraph::opset1::AvgPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);
    parent->set_friendly_name("avgPool");

    if (additionalLayer == "maxpool") {
        parent = std::make_shared<ngraph::opset1::MaxPool>(
            parent,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
        parent->set_friendly_name("maxPool");
    }

    if (addFQ) {
        parent = ngraph::builder::makeFakeQuantize(parent, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
        parent->set_friendly_name("lastFakeQuantize");
    }

    const size_t outputChannels = 6ul;
    const size_t inputChannels = 3ul;
    const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        std::make_shared<opset1::Constant>(element::f32, shape, std::vector<float>(1.f, ngraph::shape_size(shape))),
        precision,
        255,
        {outputChannels, 1, 1, 1},
        std::vector<float>(outputChannels, -1.27f),
        std::vector<float>(outputChannels, 1.27f),
        std::vector<float>(outputChannels, -1.27f),
        std::vector<float>(outputChannels, 1.27f));
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    parent = std::make_shared<ngraph::opset1::Convolution>(
        ngraph::op::TemporaryReplaceOutputType(parent, precision).get(),
        ngraph::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    //parent = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
    //    convolutionOriginal,
    //    std::vector<element::Type>{ precision, precision },
    //    std::vector<element::Type>{ precision });
    parent->set_friendly_name("convolution");

    parent->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MarkupAvgPoolPrecisions");
}

std::shared_ptr<ngraph::Function> MarkupAvgPoolPrecisionsFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, ngraph::Shape(inputShape));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        fakeQuantize,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(avgPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MarkupAvgPoolPrecisions");
}

std::shared_ptr<ngraph::Function> MarkupAvgPoolPrecisionsFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape(inputShape));

    const auto deqBefore = makeDequantization(input, dequantizationBefore);
    auto outPrecision = precisionAfterOperation;
    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::AvgPool>>(
        opset1::AvgPool(
            deqBefore,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            true,
            op::RoundingType::FLOOR),
        outPrecision);

    std::shared_ptr<Node> lastLayer = avgPool;
    if (additionalLayer == "maxpool") {
        lastLayer = std::make_shared<ngraph::opset1::MaxPool>(
            lastLayer,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
    }
    auto deqAfterStructure = dequantizationAfter;
    deqAfterStructure.multiply.outPrecision = precision;
    lastLayer = makeDequantization(lastLayer, deqAfterStructure);

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastLayer) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MarkupAvgPoolPrecisions");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
