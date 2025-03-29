// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/precision_propagation.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> PrecisionPropagationFunction::getOriginalWithNeighbors(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const DequantizationOperations::Convert& convert1,
    const DequantizationOperations& dequantization1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations::Convert& convert2,
    const DequantizationOperations& dequantization2,
    const FakeQuantizeOnData& fqOnData3,
    const DequantizationOperations::Convert& convert3,
    const DequantizationOperations& dequantization3) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    std::shared_ptr<Node> parent1;
    {
        input1->set_friendly_name("input1");
        const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
        fakeQuantize1->set_friendly_name("fakeQuantize1");
        parent1 = fakeQuantize1;

        if (!convert1.empty()) {
            parent1 = std::make_shared<ov::opset1::Convert>(parent1, convert1.outPrecision);
        }
        if (!dequantization1.empty()) {
            parent1 = makeDequantization(parent1, dequantization1);
        }
    }

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    std::shared_ptr<Node> parent2;
    {
        input2->set_friendly_name("input2");
        const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
        fakeQuantize2->set_friendly_name("fakeQuantize2");
        parent2 = fakeQuantize2;

        if (!convert2.empty()) {
            parent2 = std::make_shared<ov::opset1::Convert>(parent2, convert2.outPrecision);
        }
        if (!dequantization2.empty()) {
            parent2 = makeDequantization(parent2, dequantization2);
        }
    }

    const auto input3 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    std::shared_ptr<Node> parent3;
    {
        input3->set_friendly_name("input3");
        const auto fakeQuantize3 = makeFakeQuantize(input3, precision, fqOnData3);
        fakeQuantize3->set_friendly_name("fakeQuantize3");
        parent3 = fakeQuantize3;

        if (!convert3.empty()) {
            parent3 = std::make_shared<ov::opset1::Convert>(parent3, convert3.outPrecision);
        }
        if (!dequantization3.empty()) {
            parent3 = makeDequantization(parent3, dequantization3);
        }
    }

    const auto concat1 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { parent1->output(0), parent2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = "concat1";

    const auto concat2 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { parent2->output(0), parent3->output(0) },
        1ull);
    concat2->set_friendly_name("concat2");

    auto& rtInfo2 = concat2->get_rt_info();
    rtInfo2["Variant::std::string"] = "concat2";

    std::shared_ptr<ov::Node> result1 = concat1;
    std::shared_ptr<ov::Node> result2 = concat2;
    {
        const std::vector<size_t> kernel = { 3, 3 };
        const std::vector<size_t> stride = { 1, 1 };
        const std::vector<size_t> padBegin = { 0, 0 };
        const std::vector<size_t> padEnd = { 0, 0 };
        const ov::op::PadType padType = ov::op::PadType::NOTSET;
        const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

        result2 = std::make_shared<ov::opset1::MaxPool>(
            result2,
            stride,
            padBegin,
            padEnd,
            kernel,
            roundingType,
            padType);
        result2->set_friendly_name("MaxPool");

        const size_t outputChannels = 9ul;
        const size_t inputChannels = 6ul;
        const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
        const auto fakeQuantizeOnWeights = ov::test::utils::make_fake_quantize(
            std::make_shared<ov::opset1::Constant>(ov::element::f32,
                                                   shape,
                                                   std::vector<float>(ov::shape_size(shape), 1.f)),
            precision,
            255,
            {outputChannels, 1, 1, 1},
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f),
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f));
        fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

        result2 = std::make_shared<ov::opset1::Convolution>(
            ov::op::TemporaryReplaceOutputType(result2, precision).get(),
            ov::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });

        result2->set_friendly_name("convolution");
    }

    const ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(result1),
        std::make_shared<ov::opset1::Result>(result2)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector { input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ov::Model> PrecisionPropagationFunction::getReferenceWithNeighbors(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const auto input3 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input3->set_friendly_name("input3");

    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
    fakeQuantize3->set_friendly_name("fakeQuantize3");
    const auto deqBefore3 = makeDequantization(fakeQuantize3, dequantizationBefore);

    const auto concat1 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { deqBefore1, deqBefore2 },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = "concat1";

    const auto concat2 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { deqBefore2, deqBefore3 },
        1ull);
    concat2->set_friendly_name("concat2");

    auto& rtInfo2 = concat2->get_rt_info();
    rtInfo2["Variant::std::string"] = "concat2";

    std::shared_ptr<ov::Node> result1 = concat1;
    std::shared_ptr<ov::Node> result2 = concat2;
    {
        const std::vector<size_t> kernel = { 3, 3 };
        const std::vector<size_t> stride = { 1, 1 };
        const std::vector<size_t> padBegin = { 0, 0 };
        const std::vector<size_t> padEnd = { 0, 0 };
        const ov::op::PadType padType = ov::op::PadType::NOTSET;
        const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

        result2 = std::make_shared<ov::opset1::MaxPool>(
            result2,
            stride,
            padBegin,
            padEnd,
            kernel,
            roundingType,
            padType);
        result2->set_friendly_name("MaxPool");

        const size_t outputChannels = 9ul;
        const size_t inputChannels = 6ul;

        {
            const auto shape = Shape{ 1, inputChannels, 1, 1 };
            std::shared_ptr<Node> subtractConst =
                std::make_shared<ov::opset1::Constant>(ov::element::u8,
                                                       shape,
                                                       std::vector<float>(ov::shape_size(shape), 128.f));

            auto subtract = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
                std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
                std::vector<ov::element::Type>{ov::element::f32},
                ov::op::TemporaryReplaceOutputType(result2, ov::element::f32).get(),
                ov::op::TemporaryReplaceOutputType(subtractConst, ov::element::f32).get());
            result2 = subtract;
        }

        const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
        const auto fakeQuantizeOnWeights =
            std::make_shared<ov::opset1::Constant>(ov::element::i8,
                                                   shape,
                                                   std::vector<float>(ov::shape_size(shape), 100.f));
        fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

        result2 = std::make_shared<ov::opset1::Convolution>(
            ov::op::TemporaryReplaceOutputType(result2, precision).get(),
            ov::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });

        result2->set_friendly_name("convolution");
    }

    const std::shared_ptr<ov::Node> lastDequantization1 = makeDequantization(result1, dequantizationOperations1);
    lastDequantization1->set_friendly_name("concat1");

    const std::shared_ptr<ov::Node> lastDequantization2 = makeDequantization(result2, dequantizationOperations2);
    lastDequantization2->set_friendly_name("convolution");

    const ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(lastDequantization1),
        std::make_shared<ov::opset1::Result>(lastDequantization2)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector { input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<Node> PrecisionPropagationFunction::makeMaxPool(const ov::Output<Node>& parent, const std::vector<size_t>& kernel) {
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;
    const auto pooling = std::make_shared<ov::opset1::MaxPool>(
        parent,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    return pooling;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
