// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/convolution.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ov_ops/type_relaxed.hpp>
#include "ov_models/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/quantization_granularity_attribute.hpp"

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "low_precision/network_helper.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginal(
    const ngraph::element::Type netPrecision,
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
    std::shared_ptr<ngraph::opset1::Constant> weights,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const bool transposeOnData,
    const bool transposeOnInputLow,
    const bool transposeOnInputHigh,
    const bool transposeOnOutputLow,
    const bool transposeOnOutputHigh) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    auto dequantizationStructure = dequantizationOnActivations;
    dequantizationStructure.multiply.outPrecision = netPrecision;
    const auto dequantization = makeDequantization(input, dequantizationStructure);

    bool channelsIsDynamic = inputShape.rank().is_dynamic() || inputShape[1].is_dynamic();

    const size_t inputChannelsCount = !channelsIsDynamic ? inputShape[1].get_length() : 3ul;
    const size_t outputChannelsCount = 2 * inputChannelsCount;

    if ((weights->cast_vector<float>().size() != 1ul) && (weights->cast_vector<float>().size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    if (weights->cast_vector<float>().size() == 1ul) {
        auto targetShape = ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        weights = ov::as_type_ptr<ngraph::opset1::Constant>(fold<ngraph::opset1::Broadcast>(
            weights, op::Constant::create(ngraph::element::i64, Shape{ targetShape.size() }, targetShape)));
    }

    std::shared_ptr<Node> convertedWeights;
    if (dequantizationOnWeights.empty()) {
        const auto convertOnWeights = std::make_shared<opset1::Convert>(weights, netPrecision);
        OutputVector convertedOutput(1);
        convertOnWeights->constant_fold(convertedOutput, convertOnWeights->input_values());
        convertedWeights = convertedOutput[0].get_node_shared_ptr();
    } else {
        convertedWeights = weights;
    }

    const std::shared_ptr<ngraph::Node> constant = ngraph::opset1::Constant::create(ngraph::element::u64, ngraph::Shape{4}, {0, 1, 2, 3});
    std::shared_ptr<Node> onWeights;
    if (fqOnWeights.empty()) {
        onWeights = dequantizationOnWeights.empty() ? convertedWeights : makeDequantization(convertedWeights, dequantizationOnWeights);
    } else {
        onWeights = std::make_shared<opset1::FakeQuantize>(
            transposeOnData ? std::make_shared<opset1::Transpose>(convertedWeights, constant) : convertedWeights,
            transposeOnInputLow ?
                std::make_shared<opset1::Transpose>(
                    makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.inputLowValues, fqOnWeights.inputLowValues.empty()),
                    constant->clone_with_new_inputs({})) :
                makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.inputLowValues, fqOnWeights.inputLowValues.empty()),
            transposeOnInputHigh ?
                std::make_shared<opset1::Transpose>(
                    makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.inputHighValues, fqOnWeights.inputHighValues.empty()),
                    constant->clone_with_new_inputs({})) :
                makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.inputHighValues, fqOnWeights.inputHighValues.empty()),
            transposeOnOutputLow ?
                std::make_shared<opset1::Transpose>(
                    makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.outputLowValues, fqOnWeights.outputLowValues.empty()),
                    constant->clone_with_new_inputs({})) :
                makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.outputLowValues, fqOnWeights.outputLowValues.empty()),
            transposeOnOutputHigh ?
                std::make_shared<opset1::Transpose>(
                    makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.outputHighValues, fqOnWeights.outputHighValues.empty()),
                    constant->clone_with_new_inputs({})) :
                makeConstant(netPrecision, fqOnWeights.constantShape, fqOnWeights.outputHighValues, fqOnWeights.outputHighValues.empty()),
            fqOnWeights.quantizationLevel);
    }

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ov::op::TemporaryReplaceOutputType(dequantization, netPrecision).get(),
        ov::op::TemporaryReplaceOutputType(onWeights, netPrecision).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ netPrecision, netPrecision },
        std::vector<element::Type>{ netPrecision });
    convolution->set_friendly_name("output");
    auto& rtInfo = convolution->get_rt_info();
    rtInfo["Variant::std::string"] = "convolution";

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginalWithIncorrectWeights(
    const ngraph::Shape& inputShape,
    ngraph::element::Type precision,
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ngraph::builder::subgraph::DequantizationOperations dequantization,
    bool isCorrect) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto deq = makeDequantization(input, dequantization);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];
    const auto weights = ngraph::opset1::Constant::create(
        ngraph::element::f32,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    const auto fqOnWeights = fakeQuantizeOnWeights.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            weights, ngraph::element::f32, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    const auto subtract = isCorrect ? nullptr : std::make_shared<opset1::Subtract>(fqOnWeights,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, Shape{1, 1, 1, 1}, 3.0f));

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        deq,
        isCorrect ? fqOnWeights : subtract,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getOriginalWithIncorrectWeights(
    const ngraph::PartialShape& inputShape,
    ngraph::element::Type precision,
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
    bool isCorrect) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto fqOnData = fakeQuantizeOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const size_t inputChannelsCount = inputShape[1].get_length();
    const size_t outputChannelsCount = 2 * inputShape[1].get_length();
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

    const auto subtract = isCorrect ? nullptr : std::make_shared<opset1::Subtract>(fqOnWeights,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, 3.0f));

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
    ngraph::element::Type inputPrecision,
    ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
    ngraph::element::Type weightsPrecision,
    std::vector<float> weightsValues,
    ngraph::builder::subgraph::DequantizationOperations dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    const std::shared_ptr<ngraph::Node> weights = ngraph::opset1::Constant::create(
        weightsPrecision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ov::op::TemporaryReplaceOutputType(deqBefore, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weights, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

    const auto deqAfter = makeDequantization(convolution, dequantizationAfter);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(deqAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::getReference(
    const ngraph::element::Type netPrecision,
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ngraph::opset1::Constant> weights,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const ngraph::element::Type precisionAfterDequantization) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    auto dequantizationBeforeStructure = dequantizationBefore;
    dequantizationBeforeStructure.multiply.outPrecision = netPrecision;
    const auto deqBefore = makeDequantization(input, dequantizationBeforeStructure);

    bool channelsIsDynamic = inputShape.rank().is_dynamic() || inputShape[1].is_dynamic();

    const size_t inputChannelsCount = !channelsIsDynamic ? inputShape[1].get_length() : 3ul;
    const size_t outputChannelsCount = 2 * inputChannelsCount;

    if ((weights->cast_vector<float>().size() != 1ul) && (weights->cast_vector<float>().size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    if (weights->cast_vector<float>().size() == 1ul) {
        auto targetShape = ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        weights = ov::as_type_ptr<ngraph::opset1::Constant>(fold<ngraph::opset1::Broadcast>(
            weights, op::Constant::create(ngraph::element::i64, Shape{ targetShape.size() }, targetShape)));
    }

    const auto convertOnWeights = std::make_shared<opset1::Convert>(weights, netPrecision);
    OutputVector convertedOutput(1);
    convertOnWeights->constant_fold(convertedOutput, convertOnWeights->input_values());
    const auto convertedWeights = convertedOutput[0].get_node_shared_ptr();

    std::shared_ptr<ngraph::Node> onWeights = fakeQuantizeOnWeights.empty() ?
        (weights->get_output_element_type(0).is_real() ?
            convertedWeights :
            std::dynamic_pointer_cast<ngraph::Node>(weights)) :
        ngraph::builder::makeFakeQuantize(
            convertedWeights->output(0),
            netPrecision,
            fakeQuantizeOnWeights.quantizationLevel,
            fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues,
            fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues,
            fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ov::op::TemporaryReplaceOutputType(deqBefore, netPrecision).get(),
        ov::op::TemporaryReplaceOutputType(onWeights, netPrecision).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ netPrecision, netPrecision },
        std::vector<element::Type>{ netPrecision });

    if (!dequantizationAfter.empty()) {
        ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(convolution,
                                                                                      precisionAfterOperation);
    }
    auto& rtInfo = convolution->get_rt_info();
    rtInfo["Variant::std::string"] = "convolution";

    auto dequantizationStructure = dequantizationAfter;
    dequantizationStructure.multiply.outPrecision = netPrecision;
    const auto deqAfter = makeDequantization(convolution, dequantizationStructure);
    deqAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(deqAfter) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ngraph::Function> ConvolutionFunction::get(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precision,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& weightsValues,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
    const std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>& restrictions) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const std::shared_ptr<ngraph::opset1::FakeQuantize> fqOnData = ov::as_type_ptr<ngraph::opset1::FakeQuantize>(ngraph::builder::makeFakeQuantize(
        input,
        precision,
        fakeQuantizeOnData.quantizationLevel,
        fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues,
        fakeQuantizeOnData.inputHighValues,
        fakeQuantizeOnData.outputLowValues,
        fakeQuantizeOnData.outputHighValues));

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];
    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    const std::shared_ptr<ngraph::Node> parentOnData = fakeQuantizeOnData.empty() ? std::dynamic_pointer_cast<ngraph::Node>(input) : fqOnData;

    const std::shared_ptr<ngraph::Node> weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    const std::shared_ptr<ngraph::Node> parentOnWeights = fakeQuantizeOnWeights.empty() ?
        weights :
        ngraph::builder::makeFakeQuantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal = ngraph::opset1::Convolution(
        ov::op::TemporaryReplaceOutputType(parentOnData, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(parentOnWeights, element::f32).get(),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    const std::shared_ptr<ngraph::opset1::Convolution> convolution = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Convolution>>(
        convolutionOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});
    convolution->set_friendly_name("convolution");

    for (const auto& r : restrictions) {
        for (const auto& restrictedPort : r.restrictions) {
            auto& rt = convolution->input(restrictedPort.port).get_rt_info();
            rt[ov::QuantizationGranularityAttribute::get_type_info_static()] = ov::QuantizationGranularityAttribute(restrictedPort.granularity);
        }
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ConvolutionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
