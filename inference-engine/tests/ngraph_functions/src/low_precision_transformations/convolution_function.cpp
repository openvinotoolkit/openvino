// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_weights.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "transformations/low_precision/common/dequantization_op.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph::pass::low_precision;

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

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<DequantizationConvert>(parent, precision);
    parent = convert;

    if (!actualValues.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<DequantizationSubtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(precision, Shape({ actualValues.subtractValues.size() }), actualValues.subtractValues));
        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<DequantizationMultiply>(
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
    convolution->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginalWithIncorrectWeights(
    const ngraph::Shape& inputShape,
    ngraph::element::Type precision,
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
    bool isCorrect) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fqOnData = fakeQuantizeOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];
    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    const auto fqOnWeights = fakeQuantizeOnWeights.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    const auto subtract = isCorrect ? nullptr : std::make_shared<DequantizationSubtract>(fqOnWeights,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, Shape{1, 1, 1, 1}, 3.0f));

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        fakeQuantizeOnData.empty() ? input : fqOnData,
        isCorrect ? fqOnWeights : subtract,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getReferenceWithIncorrectWeights(
    const ngraph::Shape& inputShape,
    ngraph::element::Type precision,
    ngraph::element::Type dataPrecision,
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
    ngraph::element::Type weightsPrecision,
    std::vector<float> weightsValues,
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter,
    bool isCorrect) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<ngraph::opset1::FakeQuantize> fqOnData = as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input,
        precision,
        fakeQuantizeOnData.quantizationLevel,
        fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues,
        fakeQuantizeOnData.inputHighValues,
        fakeQuantizeOnData.outputLowValues,
        fakeQuantizeOnData.outputHighValues));

    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fqOnData, dataPrecision);

    const auto deqBefore = dequantizationBefore.empty() ? nullptr : makeDequantization(fqOnData, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        THROW_IE_EXCEPTION << "unexpected actual weights values size";
    }

    const std::shared_ptr<ngraph::Node> weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    const auto fqOnWeights = fakeQuantizeOnWeights.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    const auto subtract = isCorrect ? nullptr : std::make_shared<DequantizationSubtract>(fqOnWeights,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, 3.0f));

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ngraph::op::TemporaryReplaceOutputType(dequantizationBefore.empty() ? fqOnData : deqBefore, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(isCorrect ? weights : subtract, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

    std::shared_ptr<ngraph::Node> multiply;
    if (!dequantizationAfter.multiply.empty()) {
        ngraph::Shape constShape = isCorrect ? Shape{ 1, 1, 1 } : Shape{ 1, 1, 1, 1 };
        multiply = std::make_shared<DequantizationMultiply>(convolution,
            std::make_shared<ngraph::opset1::Constant>(precision, constShape, dequantizationAfter.multiply.values[0]));
    }

    replace_node(fqOnData->get_input_node_shared_ptr(3),
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{}, fakeQuantizeOnData.outputLowValues[0]));

    replace_node(fqOnData->get_input_node_shared_ptr(4),
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{}, fakeQuantizeOnData.outputHighValues[0]));

    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fqOnData, dataPrecision);

    if (!dequantizationBefore.multiply.empty()) {
        ngraph::Shape constShape = isCorrect ? Shape{ 1, 1, 1 } : Shape{ 1, 1, 1, 1 };
        replace_node(
            deqBefore->get_input_node_shared_ptr(1),
            std::make_shared<ngraph::opset1::Constant>(precision, constShape, dequantizationBefore.multiply.values[0]));
    }

    if (isCorrect) {
        replace_node(
            weights,
            ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(weights, weightsPrecision));
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationAfter.empty() ? convolution : multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
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
        subtract = std::make_shared<ngraph::op::TypeRelaxed<DequantizationSubtract>>(
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

    std::shared_ptr<ngraph::Node> onWeights = expectedValues.fakeQuantizeOnWeights.empty() ?
        std::dynamic_pointer_cast<ngraph::Node>(weights) :
        ngraph::builder::makeFakeQuantize(
            weights->output(0),
            precision,
            expectedValues.fakeQuantizeOnWeights.quantizationLevel,
            expectedValues.fakeQuantizeOnWeights.constantShape,
            expectedValues.fakeQuantizeOnWeights.inputLowValues,
            expectedValues.fakeQuantizeOnWeights.inputHighValues,
            expectedValues.fakeQuantizeOnWeights.outputLowValues,
            expectedValues.fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(onWeights, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

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
    multiply->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
