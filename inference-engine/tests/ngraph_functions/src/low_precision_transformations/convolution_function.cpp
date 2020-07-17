// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
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
        std::make_shared<ngraph::opset1::Constant>(precision, Shape({ actualValues.mutliplyValues.size() }), actualValues.mutliplyValues));
    parent = multiply;


    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((actualValues.weightsValues.size() != 1ul) && (actualValues.weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        THROW_IE_EXCEPTION << "unexpected actual weights values size";
    }

    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        actualValues.weightsValues.size() == 1ul ?
            std::vector<float>(outputChannelsCount * inputChannelsCount, actualValues.weightsValues[0]) :
            actualValues.weightsValues);

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        parent,
        actualValues.fakeQuantizeOnWeights.empty() ? weights->output(0) :
        ngraph::builder::makeFakeQuantize(
            weights, precision,
            actualValues.fakeQuantizeOnWeights.quantizationLevel,
            actualValues.fakeQuantizeOnWeights.constantShape,
            actualValues.fakeQuantizeOnWeights.inputLowValues,
            actualValues.fakeQuantizeOnWeights.inputHighValues,
            actualValues.fakeQuantizeOnWeights.outputLowValues,
            actualValues.fakeQuantizeOnWeights.outputHighValues),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
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
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((expectedValues.weightsValues.size() != 1ul) && (expectedValues.weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        THROW_IE_EXCEPTION << "unexpected actual weights values size";
    }

    const std::shared_ptr<ngraph::opset1::Constant> weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        expectedValues.weightsValues.size() == 1ul ?
            std::vector<float>(outputChannelsCount * inputChannelsCount, expectedValues.weightsValues[0]) :
            expectedValues.weightsValues);

    const std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
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

    if (expectedValues.mutliplyValues.size() != 1ul) {
        THROW_IE_EXCEPTION << "not supported";
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        convolution,
        std::make_shared<ngraph::opset1::Constant>(
            precision,
            Shape({ expectedValues.mutliplyValues.size(), 1, 1 }),
            expectedValues.mutliplyValues));
    parent = multiply;

    if (updatePrecisions) {
        // this is not working
        // input->set_output_type(0, expectedValues.activationPrecision, input->get_output_partial_shape(0));
        input = as_type_ptr<ngraph::opset1::Parameter>(replace_node(
            input,
            std::make_shared<ngraph::opset1::Parameter>(
                expectedValues.activationPrecision,
                ngraph::Shape(inputShape))));

        if (subtract != nullptr) {
            // this is not working
            // subtract->get_input_node_shared_ptr(1)->set_output_type(0, expectedValues.activationPrecision, subtract->get_output_partial_shape(0));
            replace_node(
                subtract->get_input_node_shared_ptr(1),
                ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(subtract->get_input_node_shared_ptr(1), expectedValues.activationPrecision));
        }

        // this is not working
        // weights->set_output_type(0, expectedValues.weightsPrecision, weights->get_output_partial_shape(0));
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
