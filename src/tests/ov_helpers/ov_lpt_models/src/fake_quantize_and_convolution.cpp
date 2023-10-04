// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "inference_engine.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

// TODO: remove, reuse mode extended method
std::shared_ptr<ngraph::Function> FakeQuantizeAndConvolutionFunction::get(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const FakeQuantizeOnWeights& fqOnWeights) {
    const auto rankLength = inputShape.rank().is_dynamic() ? 4 : inputShape.rank().get_length();
    OPENVINO_ASSERT(rankLength == 3ul || rankLength == 4ul || rankLength == 5ul, "not supported input shape rank: ", rankLength);

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = fqOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, precision, fqOnData.quantizationLevel, fqOnData.constantShape,
            fqOnData.inputLowValues, fqOnData.inputHighValues, fqOnData.outputLowValues, fqOnData.outputHighValues);
    if (fakeQuantizeOnActivations != nullptr) {
        fakeQuantizeOnActivations->set_friendly_name("fakeQuantizeOnActivations");
    }

    const size_t inputChannelsCount = inputShape[1].get_length();
    const size_t outputChannelsCount = 2 * inputShape[1].get_length();
    const auto weights = ngraph::opset1::Constant::create(
        precision,
        rankLength == 3ul ?
            (ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1}) :
            (ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 }),
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    auto maxPool = std::make_shared<opset1::MaxPool>(
        fqOnData.empty() ? input : fakeQuantizeOnActivations,
        Strides(rankLength - 2, 1ul),
        Shape(rankLength - 2, 1ul),
        Shape(rankLength - 2, 0ul),
        Shape(rankLength - 2, 2ul),
        op::RoundingType::FLOOR);
    maxPool->set_friendly_name("maxPool");

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        maxPool, //fqOnData.empty() ? input : fakeQuantizeOnActivations,
        fqOnWeights.empty() ?
            weights->output(0) :
            ngraph::builder::makeFakeQuantize(
                weights, precision, fqOnWeights.quantizationLevel, fqOnWeights.constantShape,
                fqOnWeights.inputLowValues, fqOnWeights.inputHighValues, fqOnWeights.outputLowValues, fqOnWeights.outputHighValues),
        ngraph::Strides(rankLength - 2, 1ul),
        ngraph::CoordinateDiff(rankLength - 2, 0ul),
        ngraph::CoordinateDiff(rankLength - 2, 0ul),
        ngraph::Strides(rankLength - 2, 1ul));
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeAndConvolutionFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeAndConvolutionFunction::get(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const DequantizationOperations::Convert& convertOnData,
    const DequantizationOperations& dequantizationOnData,
    const Constant& constantOnWeights,
    const FakeQuantizeOnWeights& fqOnWeights,
    const DequantizationOperations::Convert& convertOnWeights,
    const DequantizationOperations& dequantizationOnWeights,
    const DequantizationOperations& dequantizationAfter,
    const std::string operation) {
    return FakeQuantizeAndConvolutionFunction::get(
        precision,
        inputShape,
        fqOnData,
        convertOnData,
        dequantizationOnData,
        constantOnWeights,
        fqOnWeights,
        convertOnWeights,
        dequantizationOnWeights,
        {},
        {},
        {},
        {},
        dequantizationAfter,
        operation);
}

std::shared_ptr<ngraph::Function> FakeQuantizeAndConvolutionFunction::get(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const DequantizationOperations::Convert& convertOnData,
    const DequantizationOperations& dequantizationOnData,
    const Constant& constantOnWeights,
    const FakeQuantizeOnWeights& fqOnWeights,
    const DequantizationOperations::Convert& convertOnWeights,
    const DequantizationOperations& dequantizationOnWeights,
    const Reshape& reshape1,
    const DequantizationOperations::Multiply& multiply,
    const Transpose& transpose,
    const Reshape& reshape2,
    const DequantizationOperations& dequantizationAfter,
    const std::string operation,
    bool multiplyAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);

    std::shared_ptr<Node> parentOnActivation = input;
    {
        if (!fqOnData.empty()) {
            parentOnActivation = fqOnData.outputPrecision == element::undefined ?
                ngraph::builder::subgraph::makeFakeQuantize(input, precision, fqOnData) :
                ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(input, precision, fqOnData);
        }

        if (!convertOnData.empty()) {
            parentOnActivation = std::make_shared<opset1::Convert>(parentOnActivation, convertOnData.outPrecision);
        }

        if (!dequantizationOnData.empty()) {
            parentOnActivation = makeDequantization(parentOnActivation, dequantizationOnData);
        }
    }

    std::shared_ptr<Node> parentOnWeights;
    {
        const bool isDynamicChannel = inputShape.is_dynamic() || inputShape[1].is_dynamic();
        size_t numGroups = !isDynamicChannel ? inputShape[1].get_length() : 3ul;
        size_t inputChannelsCount = !isDynamicChannel ? inputShape[1].get_length() : 3ul;
        size_t outputChannelsCount = inputChannelsCount * 2;

        if (operation == "GroupConvolution") {
            inputChannelsCount /= numGroups;
            outputChannelsCount = numGroups;
        }

        const Shape shape = constantOnWeights.shapeIsDefined ? constantOnWeights.shape : ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        parentOnWeights = ngraph::opset1::Constant::create(
            constantOnWeights.outPrecision,
            shape,
            constantOnWeights.values.size() != ngraph::shape_size(shape) ?
                std::vector<float>(ngraph::shape_size(shape), constantOnWeights.values[0]) :
                constantOnWeights.values);

        if (!fqOnWeights.empty()) {
            parentOnWeights = fqOnWeights.outputPrecision == element::undefined ?
                ngraph::builder::subgraph::makeFakeQuantize(parentOnWeights, parentOnWeights->output(0).get_element_type(), fqOnWeights) :
                ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(parentOnWeights, parentOnWeights->output(0).get_element_type(), fqOnWeights);
        }

        if (!convertOnWeights.empty()) {
            parentOnWeights = std::make_shared<opset1::Convert>(parentOnWeights, convertOnWeights.outPrecision);
        }

        if (!dequantizationOnWeights.empty()) {
            parentOnWeights = makeDequantization(parentOnWeights, dequantizationOnWeights);
        }

        if (!reshape1.empty()) {
            parentOnWeights = makeReshape(parentOnWeights, reshape1);
        }

        if (!multiply.empty()) {
            parentOnWeights = makeMultiply(parentOnWeights, multiply);
        }

        if (!transpose.empty()) {
            parentOnWeights = makeTranspose(parentOnWeights, transpose);
        }

        if (!reshape2.empty()) {
            parentOnWeights = makeReshape(parentOnWeights, reshape2);
        }
    }

    std::shared_ptr<Node> lastOperation;
    if (operation == "Convolution") {
        lastOperation = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Convolution>>(
            ngraph::opset1::Convolution(
                ov::op::TemporaryReplaceOutputType(parentOnActivation, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(parentOnWeights, element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 }),
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{});
    } else if (operation == "GroupConvolution") {
        lastOperation = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::GroupConvolution>>(
            ngraph::opset1::GroupConvolution(
                ov::op::TemporaryReplaceOutputType(parentOnActivation, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(parentOnWeights, element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 }),
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{});
        if (multiplyAfter) {
            const auto& O = lastOperation->get_shape()[1];
            std::vector<float> weights_val(O, 1);
            auto constant = op::Constant::create(element::f32, Shape{O, 1, 1}, weights_val);
            lastOperation = std::make_shared<ngraph::opset1::Multiply>(lastOperation, constant);
        }
    } else {
        IE_THROW() << "unknown operation type " << operation;
    }

    if (!dequantizationAfter.empty()) {
        lastOperation->set_friendly_name("output_original");
        lastOperation = makeDequantization(lastOperation, dequantizationAfter);
        lastOperation->set_friendly_name("output");
    } else {
        lastOperation->set_friendly_name("output");
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastOperation) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeAndConvolutionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
