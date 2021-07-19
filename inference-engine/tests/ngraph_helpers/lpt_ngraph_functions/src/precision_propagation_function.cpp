// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/precision_propagation_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> PrecisionPropagationFunction::getOriginalWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const DequantizationOperations::Convert& convert1,
    const DequantizationOperations& dequantization1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations::Convert& convert2,
    const DequantizationOperations& dequantization2,
    const FakeQuantizeOnData& fqOnData3,
    const DequantizationOperations::Convert& convert3,
    const DequantizationOperations& dequantization3) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    std::shared_ptr<Node> parent1;
    {
        input1->set_friendly_name("input1");
        const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
        fakeQuantize1->set_friendly_name("fakeQuantize1");
        parent1 = fakeQuantize1;

        if (!convert1.empty()) {
            parent1 = std::make_shared<opset1::Convert>(parent1, convert1.outPrecision);
        }
        if (!dequantization1.empty()) {
            parent1 = makeDequantization(parent1, dequantization1);
        }
    }

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    std::shared_ptr<Node> parent2;
    {
        input2->set_friendly_name("input2");
        const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
        fakeQuantize2->set_friendly_name("fakeQuantize2");
        parent2 = fakeQuantize2;

        if (!convert2.empty()) {
            parent2 = std::make_shared<opset1::Convert>(parent2, convert2.outPrecision);
        }
        if (!dequantization2.empty()) {
            parent2 = makeDequantization(parent2, dequantization2);
        }
    }

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    std::shared_ptr<Node> parent3;
    {
        input3->set_friendly_name("input3");
        const auto fakeQuantize3 = makeFakeQuantize(input3, precision, fqOnData3);
        fakeQuantize3->set_friendly_name("fakeQuantize3");
        parent3 = fakeQuantize3;

        if (!convert3.empty()) {
            parent3 = std::make_shared<opset1::Convert>(parent3, convert3.outPrecision);
        }
        if (!dequantization3.empty()) {
            parent3 = makeDequantization(parent3, dequantization3);
        }
    }

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { parent1->output(0), parent2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat1");

    const auto concat2 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { parent2->output(0), parent3->output(0) },
        1ull);
    concat2->set_friendly_name("concat2");

    auto& rtInfo2 = concat2->get_rt_info();
    rtInfo2["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat2");

    std::shared_ptr<ngraph::Node> result1 = concat1;
    std::shared_ptr<ngraph::Node> result2 = concat2;
    {
        const std::vector<size_t> kernel = { 3, 3 };
        const std::vector<size_t> stride = { 1, 1 };
        const std::vector<size_t> padBegin = { 0, 0 };
        const std::vector<size_t> padEnd = { 0, 0 };
        const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
        const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

        result2 = std::make_shared<ngraph::opset1::MaxPool>(
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
        const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
            std::make_shared<opset1::Constant>(element::f32, shape, std::vector<float>(ngraph::shape_size(shape), 1.f)),
            precision,
            255,
            { outputChannels, 1, 1, 1 },
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f),
            std::vector<float>(outputChannels, -1.27f),
            std::vector<float>(outputChannels, 1.27f));
        fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

        result2 = std::make_shared<ngraph::opset1::Convolution>(
            ngraph::op::TemporaryReplaceOutputType(result2, precision).get(),
            ngraph::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });

        result2->set_friendly_name("convolution");
    }

    const ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(result1),
        std::make_shared<ngraph::opset1::Result>(result2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector { input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> PrecisionPropagationFunction::getReferenceWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input3->set_friendly_name("input3");

    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
    fakeQuantize3->set_friendly_name("fakeQuantize3");
    const auto deqBefore3 = makeDequantization(fakeQuantize3, dequantizationBefore);

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, deqBefore2 },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat1");

    const auto concat2 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore2, deqBefore3 },
        1ull);
    concat2->set_friendly_name("concat2");

    auto& rtInfo2 = concat2->get_rt_info();
    rtInfo2["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat2");

    std::shared_ptr<ngraph::Node> result1 = concat1;
    std::shared_ptr<ngraph::Node> result2 = concat2;
    {
        const std::vector<size_t> kernel = { 3, 3 };
        const std::vector<size_t> stride = { 1, 1 };
        const std::vector<size_t> padBegin = { 0, 0 };
        const std::vector<size_t> padEnd = { 0, 0 };
        const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
        const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

        result2 = std::make_shared<ngraph::opset1::MaxPool>(
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
            std::shared_ptr<Node> subtractConst = std::make_shared<ngraph::opset1::Constant>(
                element::u8,
                shape,
                std::vector<float>(ngraph::shape_size(shape), 128.f));

            auto subtract = std::make_shared<op::TypeRelaxed<ngraph::pass::low_precision::DequantizationSubtract>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{ element::f32 },
                ngraph::op::TemporaryReplaceOutputType(result2, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(subtractConst, element::f32).get());
            result2 = subtract;
        }

        const auto shape = Shape{ outputChannels, inputChannels, 1, 1 };
        const auto fakeQuantizeOnWeights = std::make_shared<opset1::Constant>(element::i8, shape, std::vector<float>(ngraph::shape_size(shape), 100.f));
        fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

        result2 = std::make_shared<ngraph::opset1::Convolution>(
            ngraph::op::TemporaryReplaceOutputType(result2, precision).get(),
            ngraph::op::TemporaryReplaceOutputType(fakeQuantizeOnWeights, precision).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });

        result2->set_friendly_name("convolution");
    }

    const std::shared_ptr<ngraph::Node> lastDequantization1 = makeDequantization(result1, dequantizationOperations1);
    lastDequantization1->set_friendly_name("concat1");

    const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(result2, dequantizationOperations2);
    lastDequantization2->set_friendly_name("convolution");

    const ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(lastDequantization2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector { input1, input2, input3 },
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<Node> PrecisionPropagationFunction::makeMaxPool(const Output<Node>& parent, const std::vector<size_t>& kernel) {
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;
    const auto pooling = std::make_shared<ngraph::opset1::MaxPool>(
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
}  // namespace ngraph
