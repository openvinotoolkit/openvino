// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <ngraph/opsets/opset1.hpp>
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/fake_quantize_and_two_output_branches_with_convolution_function.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "low_precision/network_helper.hpp"
#include "ngraph_functions/builders.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::opset1::Convolution> createConvolution(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const std::shared_ptr<Node>& parent,
    const FakeQuantizeOnWeights& fqOnWeights,
    bool typeRelaxed) {
    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];
    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    const std::shared_ptr<ngraph::opset1::Convolution> convolution = typeRelaxed ?
        std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
            std::vector<element::Type>{ element::f32, element::f32 }, std::vector<element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(fqOnWeights.empty() ?
                weights :
                ngraph::builder::makeFakeQuantize(
                    weights, precision, fqOnWeights.quantizationLevel, fqOnWeights.constantShape,
                    fqOnWeights.inputLowValues, fqOnWeights.inputHighValues, fqOnWeights.outputLowValues, fqOnWeights.outputHighValues), element::f32).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 }) :
        std::make_shared<ngraph::opset1::Convolution>(
            parent,
            fqOnWeights.empty() ? weights->output(0) :
            ngraph::builder::makeFakeQuantize(
                weights, precision, fqOnWeights.quantizationLevel, fqOnWeights.constantShape,
                fqOnWeights.inputLowValues, fqOnWeights.inputHighValues, fqOnWeights.outputLowValues, fqOnWeights.outputHighValues),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });

    return convolution;
}

std::shared_ptr<ngraph::Function> FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnActivations = values.fqOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input,
            precision,
            values.fqOnData.quantizationLevel,
            values.fqOnData.constantShape,
            values.fqOnData.inputLowValues,
            values.fqOnData.inputHighValues,
            values.fqOnData.outputLowValues,
            values.fqOnData.outputHighValues);

    const std::shared_ptr<ngraph::opset1::Convolution> convolution1 = createConvolution(
        precision,
        inputShape,
        fakeQuantizeOnActivations,
        values.fqOnWeights1,
        false);

    const std::shared_ptr<ngraph::opset1::Convolution> convolution2 = createConvolution(
        precision,
        inputShape,
        fakeQuantizeOnActivations,
        values.fqOnWeights2,
        false);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(NodeVector{ convolution1, convolution2 }, 1ul);
    ngraph::ResultVector results { std::make_shared<ngraph::opset1::Result>(concat) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ExpectedValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    auto fakeQuantizeOnActivations = values.fqOnData.empty() ?
        nullptr :
        makeFakeQuantizeTypeRelaxed(input, precision, values.fqOnData);

    const std::shared_ptr<ngraph::opset1::Convolution> convolution1 = createConvolution(
        precision,
        inputShape,
        fakeQuantizeOnActivations,
        FakeQuantizeOnWeights(),
        true);
    const std::shared_ptr<ngraph::opset1::Multiply> multiply1 = std::make_shared<ngraph::opset1::Multiply>(
        convolution1,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{1, 1, 1}, values.multiplay1Values));

    const std::shared_ptr<ngraph::opset1::Convolution> convolution2 = createConvolution(
        precision,
        inputShape,
        fakeQuantizeOnActivations,
        FakeQuantizeOnWeights(),
        true);
    const std::shared_ptr<ngraph::opset1::Multiply> multiply2 = std::make_shared<ngraph::opset1::Multiply>(
        convolution2,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{1, 1, 1}, values.multiplay2Values));

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(NodeVector{ multiply1, multiply2 }, 1ul);

    if (params.updatePrecisions) {
        // fakeQuantizeOnActivations->set_output_type(0, params.precisionsOnActivations[0], fakeQuantizeOnActivations->get_output_partial_shape(0));
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantizeOnActivations, params.precisionsOnActivations[0]);

        replace_node(
            convolution1->get_input_node_shared_ptr(1),
            ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(convolution1->get_input_node_shared_ptr(1), params.precisionsOnWeights[0]));

        replace_node(
            convolution2->get_input_node_shared_ptr(1),
            ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(convolution2->get_input_node_shared_ptr(1), params.precisionsOnWeights[0]));
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
