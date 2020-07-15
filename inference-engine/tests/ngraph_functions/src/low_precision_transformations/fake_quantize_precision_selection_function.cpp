// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fake_quantize_precision_selection_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FakeQuantizePrecisionSelectionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");
    std::shared_ptr<ngraph::Node> parent = input;

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, precision,
        values.fakeQuantizeOnData.quantizationLevel,
        values.fakeQuantizeOnData.constantShape,
        values.fakeQuantizeOnData.inputLowValues,
        values.fakeQuantizeOnData.inputHighValues,
        values.fakeQuantizeOnData.outputLowValues,
        values.fakeQuantizeOnData.outputHighValues);
    parent = fakeQuantize;

    std::shared_ptr<ngraph::Node> branch1Parent;
    {
        // branch with limitation precision operation (Convolution)
        const std::shared_ptr<ngraph::Node> pooling = as_type_ptr<ngraph::Node>(std::make_shared<ngraph::opset1::MaxPool>(
            fakeQuantize,
            Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 },
            op::RoundingType::FLOOR));
        parent = pooling;

        const size_t inputChannelsCount = inputShape[1];
        const size_t outputChannelsCount = 2 * inputShape[1];

        const auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
            std::vector<float>(outputChannelsCount * inputChannelsCount, 1.f));

        branch1Parent = std::make_shared<ngraph::opset1::Convolution>(
            parent,
            values.fakeQuantizeOnWeights.empty() ?
                weights->output(0) :
                ngraph::builder::makeFakeQuantize(
                    weights,
                    precision,
                    values.fakeQuantizeOnWeights.quantizationLevel,
                    values.fakeQuantizeOnWeights.constantShape,
                    values.fakeQuantizeOnWeights.inputLowValues,
                    values.fakeQuantizeOnWeights.inputHighValues,
                    values.fakeQuantizeOnWeights.outputLowValues,
                    values.fakeQuantizeOnWeights.outputHighValues),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    std::shared_ptr<ngraph::Node> branch2Parent;
    {
        // just another branch
        branch2Parent = std::make_shared<ngraph::opset1::AvgPool>(
            fakeQuantize,
            Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 }, true,
            op::RoundingType::FLOOR);
    }

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ branch1Parent->output(0), branch2Parent->output(0) }, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizePrecisionSelectionFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizePrecisionSelectionFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ExpectedValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantizeTypeRelaxed(
        input,
        precision,
        values.fakeQuantizeOnData.quantizationLevel,
        values.fakeQuantizeOnData.constantShape,
        values.fakeQuantizeOnData.inputLowValues,
        values.fakeQuantizeOnData.inputHighValues,
        values.fakeQuantizeOnData.outputLowValues,
        values.fakeQuantizeOnData.outputHighValues);


    // branch with limitation precision operation (Convolution)
    const bool isTransparentPrecisionOperationBeforeLimited = true;
    std::shared_ptr<ngraph::Node> branch1Pooling = isTransparentPrecisionOperationBeforeLimited ?
        as_type_ptr<ngraph::Node>(std::make_shared<op::TypeRelaxed<ngraph::opset1::MaxPool>>(
            fakeQuantize,
            Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 },
            op::RoundingType::FLOOR)) :
        std::make_shared<op::TypeRelaxed<ngraph::opset1::AvgPool>>(
            fakeQuantize,
            Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 }, true,
            op::RoundingType::FLOOR);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, -126.f));

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        branch1Pooling,
        values.fakeQuantizeOnWeights.empty() ?
            weights->output(0) :
            ngraph::builder::makeFakeQuantize(
                weights,
                precision,
                values.fakeQuantizeOnWeights.quantizationLevel,
                values.fakeQuantizeOnWeights.constantShape,
                values.fakeQuantizeOnWeights.inputLowValues,
                values.fakeQuantizeOnWeights.inputHighValues,
                values.fakeQuantizeOnWeights.outputLowValues,
                values.fakeQuantizeOnWeights.outputHighValues),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::Multiply> branch1Multiply = std::make_shared<ngraph::opset1::Multiply>(
        convolution,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape({1, 1, 1}), std::vector<float>({ 0.0001f })));


    // just another branch
    std::shared_ptr<ngraph::opset1::AvgPool> branch2Pooling = std::make_shared<op::TypeRelaxed<ngraph::opset1::AvgPool>>(
        fakeQuantize,
        Strides{ 1, 1 }, Shape{ 1, 1 }, Shape{ 0, 0 }, Shape{ 2, 2 }, true,
        op::RoundingType::FLOOR);

    const std::shared_ptr<ngraph::Node> branch2Multiply = std::make_shared<ngraph::opset1::Multiply>(
        branch2Pooling,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape({}), std::vector<float>({0.01f})));

    if (values.fakeQuantizeOnDataOutPrecision != precision) {
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize, values.fakeQuantizeOnDataOutPrecision);

        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(branch1Pooling, values.fakeQuantizeOnDataOutPrecision);

        if (values.fakeQuantizeOnWeights.empty()) {
            replace_node(
                weights,
                ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(weights, ngraph::element::i8));
        }

        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(branch2Pooling, precision);
    }


    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ branch1Multiply->output(0), branch2Multiply->output(0) }, 1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizePrecisionSelectionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
