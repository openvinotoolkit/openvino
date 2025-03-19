// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/align_concat_quantization_parameters.hpp"

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> AlignConcatQuantizationParametersFunction::getOriginal(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));
    std::shared_ptr<ov::Node> parent1 = input1;
    {
        parent1 = ov::test::utils::make_fake_quantize(input1, precision, 256, {}, { -1.28 }, { 1.27 }, { -1.28 }, { 1.27 });
        parent1->set_friendly_name("fakeQuantizeOnActivations1");

        parent1 = std::make_shared<ov::opset1::AvgPool>(
            parent1,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            true,
            ov::op::RoundingType::FLOOR);
        parent1->set_friendly_name("avgPool1");

        if (additionalLayer == "maxpool") {
            parent1 = std::make_shared<ov::opset1::MaxPool>(
                parent1,
                Strides{ 1, 1 },
                Shape{ 1, 1 },
                Shape{ 0, 0 },
                Shape{ 2, 2 },
                ov::op::RoundingType::FLOOR);
            parent1->set_friendly_name("maxPool1");
        }

        if (addFQ) {
            parent1 = ov::test::utils::make_fake_quantize(parent1, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
            parent1->set_friendly_name("lastFakeQuantize1");
        }
    }

    const auto input2 = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));
    std::shared_ptr<ov::Node> parent2 = input2;
    {
        parent2 = ov::test::utils::make_fake_quantize(input1, precision, 256, {}, { -1.28f / 2.f }, { 1.27f / 2.f }, { -1.28f / 2.f }, { 1.27f / 2.f });
        parent2->set_friendly_name("fakeQuantizeOnActivations2");

        parent2 = std::make_shared<ov::opset1::AvgPool>(
            parent2,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            true,
            ov::op::RoundingType::FLOOR);
        parent2->set_friendly_name("avgPool2");

        if (additionalLayer == "maxpool") {
            parent2 = std::make_shared<ov::opset1::MaxPool>(
                parent2,
                Strides{ 1, 1 },
                Shape{ 1, 1 },
                Shape{ 0, 0 },
                Shape{ 2, 2 },
                ov::op::RoundingType::FLOOR);
            parent2->set_friendly_name("maxPool2");
        }

        if (addFQ) {
            parent2 = ov::test::utils::make_fake_quantize(parent1, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
            parent2->set_friendly_name("lastFakeQuantize2");
        }
    }
    auto parent = std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::opset1::Concat>(ov::OutputVector{ parent1, parent2 }, 1));
    parent->set_friendly_name("concat");

    {
        const size_t outputChannels = 9ul;
        const size_t inputChannels = 6ul;
        const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
        const auto fakeQuantizeOnWeights = ov::test::utils::make_fake_quantize(
            std::make_shared<ov::opset1::Constant>(ov::element::f32,
                                                   shape,
                                                   std::vector<float>(1.f, ov::shape_size(shape))),
            precision,
            255,
            {outputChannels, 1, 1, 1},
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f),
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f));
        fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

        parent = std::make_shared<ov::opset1::Convolution>(
            ov::op::TemporaryReplaceOutputType(parent, precision).get(),
            ov::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });

        parent->set_friendly_name("convolution");
    }

    parent->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input1, input2 }, "AlignConcatQuantizationParameters");
}

std::shared_ptr<ov::Model> AlignConcatQuantizationParametersFunction::getReference(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));
    std::shared_ptr<ov::Node> parent1 = input1;
    {
        FakeQuantizeOnData onData = { 256, {}, { -1.28f }, { 1.27f }, { 0.f }, { 255.f }, ov::element::u8};
        parent1 = makeFakeQuantizeTypeRelaxed(input1, ov::element::f32, onData);
        ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(parent1, ov::element::u8);
        parent1->set_friendly_name("fakeQuantizeOnActivations1");

        parent1 = std::make_shared<ov::opset1::AvgPool>(
            parent1,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            true,
            ov::op::RoundingType::FLOOR);
        parent1->set_friendly_name("avgPool1");

        if (additionalLayer == "maxpool") {
            parent1 = std::make_shared<ov::opset1::MaxPool>(
                parent1,
                Strides{ 1, 1 },
                Shape{ 1, 1 },
                Shape{ 0, 0 },
                Shape{ 2, 2 },
                ov::op::RoundingType::FLOOR);
            parent1->set_friendly_name("maxPool1");
        }

        if (addFQ) {
            parent1 = ov::test::utils::make_fake_quantize(parent1, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
            parent1->set_friendly_name("lastFakeQuantize1");
        }
    }

    const auto input2 = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));
    std::shared_ptr<ov::Node> parent2 = input2;
    {
        FakeQuantizeOnData onData = {256, {}, {-0.64f}, {0.635f}, {64.f}, {192.f}, ov::element::u8};
        parent2 = makeFakeQuantizeTypeRelaxed(input2, ov::element::f32, onData);
        ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(parent2, ov::element::u8);
        parent2->set_friendly_name("fakeQuantizeOnActivations2");

        parent2 = std::make_shared<ov::opset1::AvgPool>(
            parent2,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            true,
            ov::op::RoundingType::FLOOR);
        parent2->set_friendly_name("avgPool2");

        if (additionalLayer == "maxpool") {
            parent2 = std::make_shared<ov::opset1::MaxPool>(
                parent2,
                Strides{ 1, 1 },
                Shape{ 1, 1 },
                Shape{ 0, 0 },
                Shape{ 2, 2 },
                ov::op::RoundingType::FLOOR);
            parent2->set_friendly_name("maxPool2");
        }

        if (addFQ) {
            parent2 = ov::test::utils::make_fake_quantize(parent1, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
            parent2->set_friendly_name("lastFakeQuantize2");
        }
    }
    auto parent = std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::opset1::Concat>(ov::OutputVector{ parent1, parent2 }, 1));
    parent->set_friendly_name("concat");

    if (!dequantizationBefore.empty()) {
        parent = makeDequantization(parent, dequantizationBefore);
    }

    {
        const size_t outputChannels = 9ul;
        const size_t inputChannels = 6ul;
        const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
        const auto onWeights =
            std::make_shared<ov::opset1::Constant>(ov::element::i8,
                                                   shape,
                                                   std::vector<size_t>(outputChannels * inputChannels, 127));

        parent = std::make_shared<ov::opset1::Convolution>(
            ov::op::TemporaryReplaceOutputType(parent, precision).get(),
            ov::op::TemporaryReplaceOutputType(onWeights, precision).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });

        parent->set_friendly_name("convolution");
    }

    if (!dequantizationAfter.empty()) {
        parent = makeDequantization(parent, dequantizationAfter);
    }

    parent->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input1, input2 }, "AlignConcatQuantizationParameters");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
