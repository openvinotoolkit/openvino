// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

#include "ov_lpt_models/markup_avg_pool_precisions.hpp"
#include "ov_models/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {


std::shared_ptr<Node> createConvolution(
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const std::shared_ptr<Node>& parent) {
    const size_t outputChannels = 6ul;
    const size_t inputChannels = inputShape[1];
    const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        std::make_shared<opset1::Constant>(element::f32, shape, std::vector<float>(1.f, ngraph::shape_size(shape))),
        precision,
        255,
        { outputChannels, 1, 1, 1 },
        std::vector<float>(outputChannels, -1.27f),
        std::vector<float>(outputChannels, 1.27f),
        std::vector<float>(outputChannels, -1.27f),
        std::vector<float>(outputChannels, 1.27f));
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        ov::op::TemporaryReplaceOutputType(parent, precision).get(),
        ov::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    return convolution;
}

std::shared_ptr<ngraph::Function> MarkupAvgPoolPrecisionsFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    // -1 - no Convolution, 2 - on both branches
    const int convoutionBranch,
    // -1 - no FakeQuantize, 2 - on both branches
    const int fakeQuantizeBranch) {
    std::shared_ptr<ngraph::opset1::Parameter> input1;
    std::shared_ptr<ngraph::opset1::Parameter> input2;
    std::shared_ptr<ngraph::Node> parent;
    {
        auto createBranch = [](
            const ngraph::element::Type precision,
            const std::string& additionalLayer,
            const std::shared_ptr<ngraph::Node>& parent) -> std::shared_ptr<ngraph::Node> {
            //auto deqBeforeStructure = dequantizationBefore;
            //deqBeforeStructure.multiply.outPrecision = precision;
            // const auto parent = makeDequantization(input, deqBeforeStructure);

            auto newParent = ngraph::builder::makeFakeQuantize(parent, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
            newParent->set_friendly_name("fakeQuantizeOnActivations");

            //if (additionalLayer == "maxpool") {
            //    newParent = std::make_shared<ngraph::opset1::MaxPool>(
            //        newParent,
            //        Strides{ 1, 1 },
            //        Shape{ 1, 1 },
            //        Shape{ 0, 0 },
            //        Shape{ 2, 2 },
            //        op::RoundingType::FLOOR);
            //    newParent->set_friendly_name("maxPool1");
            //}
            return newParent;
        };
        input1 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape(inputShape));
        auto parent1 = createBranch(precision, additionalLayer, input1);

        //input2 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape(inputShape));
        //auto parent2 = createBranch(precision, additionalLayer, input2);
        //
        //parent = std::make_shared<ngraph::opset1::Concat>(OutputVector{ parent1, parent2 }, 1ul);
        parent = parent1;
    }

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
        parent = std::make_shared<ngraph::opset1::MaxPool>(parent, Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 }, op::RoundingType::FLOOR);
        parent->set_friendly_name("maxPool2");
    }

    std::shared_ptr<ngraph::Node> parent1 = std::make_shared<ngraph::opset1::MaxPool>(
        parent, Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 }, op::RoundingType::FLOOR);

    std::shared_ptr<ngraph::Node> parent2 = std::make_shared<ngraph::opset1::MaxPool>(
        parent, Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 }, op::RoundingType::FLOOR);

    //if (addFQ) {
    //    parent1 = ngraph::builder::makeFakeQuantize(parent1, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    //    parent1->set_friendly_name("lastFakeQuantize1");

    //    parent2 = ngraph::builder::makeFakeQuantize(parent2, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    //    parent2->set_friendly_name("lastFakeQuantize2");
    //}

    if (convoutionBranch != -1) {
        if (convoutionBranch != 1) {
            parent1 = createConvolution(precision, inputPrecision, inputShape, parent1);
        }
        if (convoutionBranch != 0) {
            parent2 = createConvolution(precision, inputPrecision, inputShape, parent2);
        }
    }

    if (fakeQuantizeBranch != -1) {
        if (fakeQuantizeBranch != 1) {
            parent1 = ngraph::builder::makeFakeQuantize(parent1, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
            parent1->set_friendly_name("fakeQuantize1");
        }
        if (fakeQuantizeBranch != 0) {
            parent2 = ngraph::builder::makeFakeQuantize(parent2, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
            parent2->set_friendly_name("fakeQuantize2");
        }
    }

    parent2->set_friendly_name("output");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(parent1),
        std::make_shared<ngraph::opset1::Result>(parent2)
    };

    return std::make_shared<ngraph::Function>(
        results,
        (input2 == nullptr) ? ngraph::ParameterVector{ input1 } : ngraph::ParameterVector{ input1, input2 },
        "MarkupAvgPoolPrecisions");
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
    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::AvgPool>>(
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
