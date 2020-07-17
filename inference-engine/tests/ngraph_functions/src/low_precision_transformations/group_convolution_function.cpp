// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/group_convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph::opset1;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<Node> createWeightsOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const size_t inputChannelsCount,
    const size_t outputChannelsCount,
    const size_t groupCount,
    const size_t kernelSize,
    const std::vector<float>& weightsValues,
    const FakeQuantizeOnWeights& fakeQuantizeOnWeights) {
    std::shared_ptr<Node> weights;
    if (fakeQuantizeOnWeights.empty()) {
        weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
            weightsValues.size() == 1ul ?
                std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
                weightsValues);
    } else {
        const size_t inputChannelsPerGroup = inputChannelsCount / groupCount;
        const std::shared_ptr<ngraph::opset1::Constant> weightsConst = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ outputChannelsCount, inputChannelsPerGroup, kernelSize, kernelSize },
            weightsValues.size() == 1ul ?
                std::vector<float>(outputChannelsCount * kernelSize * kernelSize * inputChannelsPerGroup, weightsValues[0]) :
                weightsValues);

        const std::shared_ptr<ngraph::Node> fakeQuantize = ngraph::builder::makeFakeQuantize(
            weightsConst,
            precision,
            fakeQuantizeOnWeights.quantizationLevel,
            { outputChannelsCount, 1, 1, 1 },
            fakeQuantizeOnWeights.inputLowValues,
            fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues,
            fakeQuantizeOnWeights.outputHighValues);

        const std::shared_ptr<ngraph::opset1::Reshape> reshape = std::make_shared<ngraph::opset1::Reshape>(
            fakeQuantize,
            ngraph::opset1::Constant::create(
                element::i64,
                Shape{ 5 },
                std::vector<size_t>({ groupCount, outputChannelsCount / groupCount, inputChannelsPerGroup, 7, 7 })),
            true);

        weights = reshape;
    }

    return weights;
}

std::shared_ptr<ngraph::Function> GroupConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::Shape& outputShape,
    const size_t groupCount,
    const bool updatePrecisions,
    const ActualValues& actualValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        updatePrecisions ? actualValues.lowPrecision : precision,
        ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, precision);
    parent = convert;

    if (!actualValues.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(precision, Shape({ actualValues.subtractValues.size() }), actualValues.subtractValues));
        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(
            precision,
            actualValues.mutliplyValues.size() == 1ul ? Shape({ actualValues.mutliplyValues.size() }) : Shape({ actualValues.mutliplyValues.size(), 1ul, 1ul }),
            actualValues.mutliplyValues));
    parent = multiply;

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 7ul;

    if ((actualValues.weightsValues.size() != 1ul) && (actualValues.weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        THROW_IE_EXCEPTION << "unexpected actual weights values size";
    }

    std::shared_ptr<ngraph::Node> weights = createWeightsOriginal(
        precision,
        inputShape,
        inputChannelsCount,
        outputChannelsCount,
        groupCount,
        kernelSize,
        actualValues.weightsValues,
        actualValues.fakeQuantizeOnWeights);

    const auto convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
        parent,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "GroupConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> GroupConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::Shape& outputShape,
    const size_t groupCount,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const FakeQuantizeOnWeights& fakeQuantizeOnWeights) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));

    std::shared_ptr<ngraph::Node> fakeQuantizeOnActivations;
    if (fakeQuantizeOnData.empty()) {
        fakeQuantizeOnActivations = nullptr;
    } else {
        fakeQuantizeOnActivations = std::make_shared<ngraph::opset1::FakeQuantize>(
            input,
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.inputLowValues.size(), 1, 1 }, fakeQuantizeOnData.inputLowValues),
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.inputHighValues.size(), 1, 1 }, fakeQuantizeOnData.inputHighValues),
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.outputLowValues.size(), 1, 1 }, fakeQuantizeOnData.outputLowValues),
            std::make_shared<Constant>(precision, Shape{ 1, fakeQuantizeOnData.outputHighValues.size(), 1, 1 }, fakeQuantizeOnData.outputHighValues),
            fakeQuantizeOnData.quantizationLevel);
    }

    // TODO: pass as argument
    //const size_t groupCount = 3ul;
    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 7ul;
    const size_t inputChannelsCount = inputShape[1];

    std::vector<float> weightsValues = { 1.f };
    std::shared_ptr<ngraph::Node> weights = createWeightsOriginal(
        precision,
        inputShape,
        inputChannelsCount,
        outputChannelsCount,
        groupCount,
        kernelSize,
        weightsValues,
        fakeQuantizeOnWeights);

    const auto convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
        fakeQuantizeOnActivations == nullptr ? input : fakeQuantizeOnActivations,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "GroupConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> GroupConvolutionFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::Shape& outputShape,
    const size_t groupCount,
    const bool updatePrecisions,
    const ExpectedValues& expectedValues) {
    std::shared_ptr<ngraph::opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precision,
        ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    std::shared_ptr<ngraph::opset1::Subtract> subtract;
    if (!expectedValues.subtractValues.empty()) {
        subtract = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                precision,
                // CPU workaround
                Shape({ 1, inputShape[1], 1, 1 }),
                expectedValues.subtractValues.size() == 1ul ?
                std::vector<float>(inputShape[1], expectedValues.subtractValues[0]) :
                expectedValues.subtractValues));
        subtract->set_output_type(0, precision, subtract->get_output_partial_shape(0));
        parent = subtract;
    }

    const size_t inputChannelsCount = inputShape[1];

    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 7ul;
    const size_t inputChannelsInGroup = inputChannelsCount / groupCount;
    const size_t outputChannelsInGroup = outputChannelsCount / groupCount;

    if ((expectedValues.weightsValues.size() != 1ul) && (expectedValues.weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        THROW_IE_EXCEPTION << "unexpected actual weights values size";
    }

    const ngraph::Shape weightsShape = ngraph::Shape{ groupCount, outputChannelsInGroup, inputChannelsInGroup, kernelSize, kernelSize };
    const std::shared_ptr<ngraph::opset1::Constant> weights = ngraph::opset1::Constant::create(
        precision,
        weightsShape,
        expectedValues.weightsValues.size() == 1ul ?
            std::vector<float>(groupCount * outputChannelsInGroup * inputChannelsInGroup * kernelSize * kernelSize, expectedValues.weightsValues[0]) :
            expectedValues.weightsValues);

    const std::shared_ptr<ngraph::opset1::GroupConvolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
        parent,
        expectedValues.fakeQuantizeOnWeights.empty() ?
            weights->output(0) :
            ngraph::builder::makeFakeQuantize(
                weights->output(0),
                precision,
                expectedValues.fakeQuantizeOnWeights.quantizationLevel,
                expectedValues.fakeQuantizeOnWeights.constantShape,
                expectedValues.fakeQuantizeOnWeights.inputLowValues,
                expectedValues.fakeQuantizeOnWeights.inputHighValues,
                expectedValues.fakeQuantizeOnWeights.outputLowValues,
                expectedValues.fakeQuantizeOnWeights.outputHighValues),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        convolution,
        std::make_shared<ngraph::opset1::Constant>(
            precision,
            Shape({ expectedValues.mutliplyValues.size(), 1, 1 }),
            expectedValues.mutliplyValues));
    parent = multiply;

    if (updatePrecisions) {
        input = as_type_ptr<ngraph::opset1::Parameter>(replace_node(
            input,
            std::make_shared<ngraph::opset1::Parameter>(
                expectedValues.activationPrecision,
                ngraph::Shape(inputShape))));

        if (subtract != nullptr) {
            replace_node(
                subtract->get_input_node_shared_ptr(1),
                ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(subtract->get_input_node_shared_ptr(1), expectedValues.activationPrecision));
        }

        replace_node(
            weights,
            ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(weights, expectedValues.weightsPrecision));
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
