// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

#include "ov_lpt_models/markup_avg_pool_precisions.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {


std::shared_ptr<Node> createConvolution(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::Shape& inputShape,
    const std::shared_ptr<Node>& parent) {
    const size_t outputChannels = 6ul;
    const size_t inputChannels = inputShape[1];
    const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
    const auto fakeQuantizeOnWeights = ov::test::utils::make_fake_quantize(
        std::make_shared<ov::opset1::Constant>(ov::element::f32, shape, std::vector<float>(1.f, ov::shape_size(shape))),
        precision,
        255,
        {outputChannels, 1, 1, 1},
        std::vector<float>(outputChannels, -1.27f),
        std::vector<float>(outputChannels, 1.27f),
        std::vector<float>(outputChannels, -1.27f),
        std::vector<float>(outputChannels, 1.27f));
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    auto convolution = std::make_shared<ov::opset1::Convolution>(
        ov::op::TemporaryReplaceOutputType(parent, precision).get(),
        ov::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    return convolution;
}

std::shared_ptr<ov::Model> MarkupAvgPoolPrecisionsFunction::getOriginal(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    // -1 - no Convolution, 2 - on both branches
    const int convoutionBranch,
    // -1 - no FakeQuantize, 2 - on both branches
    const int fakeQuantizeBranch) {
    std::shared_ptr<ov::opset1::Parameter> input1;
    std::shared_ptr<ov::opset1::Parameter> input2;
    std::shared_ptr<ov::Node> parent;
    {
        auto createBranch = [](
            const ov::element::Type precision,
            const std::string& additionalLayer,
            const std::shared_ptr<ov::Node>& parent) -> std::shared_ptr<ov::Node> {
            //auto deqBeforeStructure = dequantizationBefore;
            //deqBeforeStructure.multiply.outPrecision = precision;
            // const auto parent = makeDequantization(input, deqBeforeStructure);

            auto newParent = ov::test::utils::make_fake_quantize(parent, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
            newParent->set_friendly_name("fakeQuantizeOnActivations");

            // if (additionalLayer == "maxpool") {
            //     newParent = std::make_shared<ov::opset1::MaxPool>(
            //         newParent,
            //         Strides{ 1, 1 },
            //         Shape{ 1, 1 },
            //         Shape{ 0, 0 },
            //         Shape{ 2, 2 },
            //         ov::op::RoundingType::FLOOR);
            //     newParent->set_friendly_name("maxPool1");
            // }
            return newParent;
        };
        input1 = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));
        auto parent1 = createBranch(precision, additionalLayer, input1);

        //input2 = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));
        //auto parent2 = createBranch(precision, additionalLayer, input2);
        //
        //parent = std::make_shared<ov::opset1::Concat>(OutputVector{ parent1, parent2 }, 1ul);
        parent = parent1;
    }

    parent = std::make_shared<ov::opset1::AvgPool>(parent,
                                                   Strides{1, 1},
                                                   Shape{1, 1},
                                                   Shape{0, 0},
                                                   Shape{2, 2},
                                                   true,
                                                   ov::op::RoundingType::FLOOR);
    parent->set_friendly_name("avgPool");

    if (additionalLayer == "maxpool") {
        parent = std::make_shared<ov::opset1::MaxPool>(parent,
                                                       Strides{1, 1},
                                                       Shape{1, 1},
                                                       Shape{0, 0},
                                                       Shape{2, 2},
                                                       ov::op::RoundingType::FLOOR);
        parent->set_friendly_name("maxPool2");
    }

    std::shared_ptr<ov::Node> parent1 = std::make_shared<ov::opset1::MaxPool>(parent,
                                                                              Strides{1, 1},
                                                                              Shape{1, 1},
                                                                              Shape{0, 0},
                                                                              Shape{2, 2},
                                                                              ov::op::RoundingType::FLOOR);

    std::shared_ptr<ov::Node> parent2 = std::make_shared<ov::opset1::MaxPool>(parent,
                                                                              Strides{1, 1},
                                                                              Shape{1, 1},
                                                                              Shape{0, 0},
                                                                              Shape{2, 2},
                                                                              ov::op::RoundingType::FLOOR);

    //if (addFQ) {
    //    parent1 = ov::test::utils::make_fake_quantize(parent1, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    //    parent1->set_friendly_name("lastFakeQuantize1");

    //    parent2 = ov::test::utils::make_fake_quantize(parent2, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
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
            parent1 = ov::test::utils::make_fake_quantize(parent1, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
            parent1->set_friendly_name("fakeQuantize1");
        }
        if (fakeQuantizeBranch != 0) {
            parent2 = ov::test::utils::make_fake_quantize(parent2, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
            parent2->set_friendly_name("fakeQuantize2");
        }
    }

    parent2->set_friendly_name("output");

    ov::ResultVector results{
        std::make_shared<ov::opset1::Result>(parent1),
        std::make_shared<ov::opset1::Result>(parent2)
    };

    return std::make_shared<ov::Model>(
        results,
        (input2 == nullptr) ? ov::ParameterVector{ input1 } : ov::ParameterVector{ input1, input2 },
        "MarkupAvgPoolPrecisions");
}

std::shared_ptr<ov::Model> MarkupAvgPoolPrecisionsFunction::getOriginal(
    const ov::element::Type originalFunctionPrecision,
    const ov::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(originalFunctionPrecision, ov::Shape(inputShape));

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ov::Node> avgPool = std::make_shared<ov::opset1::AvgPool>(fakeQuantize,
                                                                                    Strides{1, 1},
                                                                                    Shape{1, 1},
                                                                                    Shape{0, 0},
                                                                                    Shape{2, 2},
                                                                                    true,
                                                                                    ov::op::RoundingType::FLOOR);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(avgPool) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MarkupAvgPoolPrecisions");
}

std::shared_ptr<ov::Model> MarkupAvgPoolPrecisionsFunction::getReference(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));

    const auto deqBefore = makeDequantization(input, dequantizationBefore);
    auto outPrecision = precisionAfterOperation;
    const std::shared_ptr<ov::Node> avgPool =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::AvgPool>>(ov::opset1::AvgPool(deqBefore,
                                                                                       Strides{1, 1},
                                                                                       Shape{1, 1},
                                                                                       Shape{0, 0},
                                                                                       Shape{2, 2},
                                                                                       true,
                                                                                       ov::op::RoundingType::FLOOR),
                                                                   outPrecision);

    std::shared_ptr<Node> lastLayer = avgPool;
    if (additionalLayer == "maxpool") {
        lastLayer = std::make_shared<ov::opset1::MaxPool>(lastLayer,
                                                          Strides{1, 1},
                                                          Shape{1, 1},
                                                          Shape{0, 0},
                                                          Shape{2, 2},
                                                          ov::op::RoundingType::FLOOR);
    }
    auto deqAfterStructure = dequantizationAfter;
    deqAfterStructure.multiply.outPrecision = precision;
    lastLayer = makeDequantization(lastLayer, deqAfterStructure);

    if (addFQ) {
        lastLayer = ov::test::utils::make_fake_quantize(
            lastLayer, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastLayer) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "MarkupAvgPoolPrecisions");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
