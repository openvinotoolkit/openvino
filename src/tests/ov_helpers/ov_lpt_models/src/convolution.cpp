// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/convolution.hpp"

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/quantization_granularity_attribute.hpp"

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "low_precision/network_helper.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> ConvolutionFunction::getOriginal(
    const ov::element::Type netPrecision,
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::DequantizationOperations& dequantizationOnActivations,
    std::shared_ptr<ov::opset1::Constant> weights,
    const ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights,
    const ov::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const bool transposeOnData,
    const bool transposeOnInputLow,
    const bool transposeOnInputHigh,
    const bool transposeOnOutputLow,
    const bool transposeOnOutputHigh) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
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
        auto targetShape = ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        weights = ov::as_type_ptr<ov::opset1::Constant>(fold<ov::opset1::Broadcast>(
            weights, ov::opset1::Constant::create(ov::element::i64, Shape{ targetShape.size() }, targetShape)));
    }

    std::shared_ptr<Node> convertedWeights;
    if (dequantizationOnWeights.empty()) {
        const auto convertOnWeights = std::make_shared<ov::opset1::Convert>(weights, netPrecision);
        OutputVector convertedOutput(1);
        convertOnWeights->constant_fold(convertedOutput, convertOnWeights->input_values());
        convertedWeights = convertedOutput[0].get_node_shared_ptr();
    } else {
        convertedWeights = weights;
    }

    const std::shared_ptr<ov::Node> constant = ov::opset1::Constant::create(ov::element::u64, ov::Shape{4}, {0, 1, 2, 3});
    std::shared_ptr<Node> onWeights;
    if (fqOnWeights.empty()) {
        onWeights = dequantizationOnWeights.empty() ? convertedWeights : makeDequantization(convertedWeights, dequantizationOnWeights);
    } else {
        onWeights = std::make_shared<ov::opset1::FakeQuantize>(
            transposeOnData ? std::make_shared<ov::opset1::Transpose>(convertedWeights, constant) : convertedWeights,
            transposeOnInputLow ?
                std::make_shared<ov::opset1::Transpose>(
                    ov::test::utils::make_constant(
                        netPrecision, fqOnWeights.constantShape, fqOnWeights.inputLowValues),
                    constant->clone_with_new_inputs({})) :
                ov::test::utils::make_constant(
                    netPrecision, fqOnWeights.constantShape, fqOnWeights.inputLowValues),
            transposeOnInputHigh ?
                std::make_shared<ov::opset1::Transpose>(
                    ov::test::utils::make_constant(
                        netPrecision, fqOnWeights.constantShape, fqOnWeights.inputHighValues),
                    constant->clone_with_new_inputs({})) :
                ov::test::utils::make_constant(
                    netPrecision, fqOnWeights.constantShape, fqOnWeights.inputHighValues),
            transposeOnOutputLow ?
                std::make_shared<ov::opset1::Transpose>(
                    ov::test::utils::make_constant(
                        netPrecision, fqOnWeights.constantShape, fqOnWeights.outputLowValues),
                    constant->clone_with_new_inputs({})) :
                ov::test::utils::make_constant(
                    netPrecision, fqOnWeights.constantShape, fqOnWeights.outputLowValues),
            transposeOnOutputHigh ?
                std::make_shared<ov::opset1::Transpose>(
                    ov::test::utils::make_constant(
                        netPrecision, fqOnWeights.constantShape, fqOnWeights.outputHighValues),
                    constant->clone_with_new_inputs({})) :
                ov::test::utils::make_constant(
                    netPrecision, fqOnWeights.constantShape, fqOnWeights.outputHighValues),
            fqOnWeights.quantizationLevel);
    }

    auto convolutionOriginal = ov::opset1::Convolution(
        ov::op::TemporaryReplaceOutputType(dequantization, netPrecision).get(),
        ov::op::TemporaryReplaceOutputType(onWeights, netPrecision).get(),
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    std::shared_ptr<ov::opset1::Convolution> convolution =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            convolutionOriginal,
            std::vector<ov::element::Type>{netPrecision, netPrecision},
            std::vector<ov::element::Type>{netPrecision});
    convolution->set_friendly_name("output");
    auto& rtInfo = convolution->get_rt_info();
    rtInfo["Variant::std::string"] = "convolution";

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolution) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ov::Model> ConvolutionFunction::getOriginalWithIncorrectWeights(
    const ov::Shape& inputShape,
    ov::element::Type precision,
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ov::builder::subgraph::DequantizationOperations dequantization,
    bool isCorrect) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    const auto deq = makeDequantization(input, dequantization);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];
    const auto weights = ov::opset1::Constant::create(
        ov::element::f32,
        ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    const auto fqOnWeights = fakeQuantizeOnWeights.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            weights, ov::element::f32, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    const auto subtract = isCorrect ? nullptr : std::make_shared<ov::opset1::Subtract>(fqOnWeights,
        std::make_shared<ov::opset1::Constant>(ov::element::f32, Shape{1, 1, 1, 1}, 3.0f));

    const auto convolution = std::make_shared<ov::opset1::Convolution>(
        deq,
        isCorrect ? fqOnWeights : subtract,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolution) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ov::Model> ConvolutionFunction::getOriginalWithIncorrectWeights(
    const ov::PartialShape& inputShape,
    ov::element::Type precision,
    ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData,
    bool isCorrect) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fqOnData = fakeQuantizeOnData.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const size_t inputChannelsCount = inputShape[1].get_length();
    const size_t outputChannelsCount = 2 * inputShape[1].get_length();
    const auto weights = ov::opset1::Constant::create(
        precision,
        ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    const auto fqOnWeights = fakeQuantizeOnWeights.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    const auto subtract = isCorrect ? nullptr : std::make_shared<ov::opset1::Subtract>(fqOnWeights,
        std::make_shared<ov::opset1::Constant>(precision, Shape{ 1, 1, 1, 1 }, 3.0f));

    const auto convolution = std::make_shared<ov::opset1::Convolution>(
        fakeQuantizeOnData.empty() ? input : fqOnData,
        isCorrect ? fqOnWeights : subtract,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolution) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ov::Model> ConvolutionFunction::getReferenceWithIncorrectWeights(
    const ov::Shape& inputShape,
    ov::element::Type inputPrecision,
    ov::builder::subgraph::DequantizationOperations dequantizationBefore,
    ov::element::Type weightsPrecision,
    std::vector<float> weightsValues,
    ov::builder::subgraph::DequantizationOperations dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, ov::Shape(inputShape));
    input->set_friendly_name("input");

    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    if ((weightsValues.size() != 1ul) && (weightsValues.size() != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    const std::shared_ptr<ov::Node> weights = ov::opset1::Constant::create(
        weightsPrecision,
        ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    auto convolutionOriginal =
        ov::opset1::Convolution(ov::op::TemporaryReplaceOutputType(deqBefore, ov::element::f32).get(),
                                ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
                                ov::Strides{1, 1},
                                ov::CoordinateDiff{0, 0},
                                ov::CoordinateDiff{0, 0},
                                ov::Strides{1, 1});

    std::shared_ptr<ov::opset1::Convolution> convolution =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            convolutionOriginal,
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{});

    const auto deqAfter = makeDequantization(convolution, dequantizationAfter);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(deqAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "IncorrectWeightsAndConvolutionFunction");
}

std::shared_ptr<ov::Model> ConvolutionFunction::getReference(
    const ov::element::Type netPrecision,
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ov::opset1::Constant> weights,
    const ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const ov::element::Type precisionAfterDequantization) {
    auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
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
        auto targetShape = ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        weights = ov::as_type_ptr<ov::opset1::Constant>(fold<ov::opset1::Broadcast>(
            weights, ov::opset1::Constant::create(ov::element::i64, Shape{ targetShape.size() }, targetShape)));
    }

    const auto convertOnWeights = std::make_shared<ov::opset1::Convert>(weights, netPrecision);
    OutputVector convertedOutput(1);
    convertOnWeights->constant_fold(convertedOutput, convertOnWeights->input_values());
    const auto convertedWeights = convertedOutput[0].get_node_shared_ptr();

    std::shared_ptr<ov::Node> onWeights = fakeQuantizeOnWeights.empty() ?
        (weights->get_output_element_type(0).is_real() ?
            convertedWeights :
            std::dynamic_pointer_cast<ov::Node>(weights)) :
        ov::test::utils::make_fake_quantize(
            convertedWeights->output(0),
            netPrecision,
            fakeQuantizeOnWeights.quantizationLevel,
            fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues,
            fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues,
            fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal = ov::opset1::Convolution(
        ov::op::TemporaryReplaceOutputType(deqBefore, netPrecision).get(),
        ov::op::TemporaryReplaceOutputType(onWeights, netPrecision).get(),
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });

    std::shared_ptr<ov::opset1::Convolution> convolution =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            convolutionOriginal,
            std::vector<ov::element::Type>{netPrecision, netPrecision},
            std::vector<ov::element::Type>{netPrecision});

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

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(deqAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ConvolutionTransformation");
}

std::shared_ptr<ov::Model> ConvolutionFunction::get(
    const ov::Shape& inputShape,
    const ov::element::Type precision,
    const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& weightsValues,
    const ov::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
    const std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>& restrictions) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input->set_friendly_name("input");

    const std::shared_ptr<ov::opset1::FakeQuantize> fqOnData = ov::as_type_ptr<ov::opset1::FakeQuantize>(ov::test::utils::make_fake_quantize(
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

    const std::shared_ptr<ov::Node> parentOnData = fakeQuantizeOnData.empty() ? std::dynamic_pointer_cast<ov::Node>(input) : fqOnData;

    const std::shared_ptr<ov::Node> weights = ov::opset1::Constant::create(
        precision,
        ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        weightsValues.size() == 1ul ?
        std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
        weightsValues);

    const std::shared_ptr<ov::Node> parentOnWeights = fakeQuantizeOnWeights.empty() ?
        weights :
        ov::test::utils::make_fake_quantize(
            weights, precision, fakeQuantizeOnWeights.quantizationLevel, fakeQuantizeOnWeights.constantShape,
            fakeQuantizeOnWeights.inputLowValues, fakeQuantizeOnWeights.inputHighValues,
            fakeQuantizeOnWeights.outputLowValues, fakeQuantizeOnWeights.outputHighValues);

    auto convolutionOriginal =
        ov::opset1::Convolution(ov::op::TemporaryReplaceOutputType(parentOnData, ov::element::f32).get(),
                                ov::op::TemporaryReplaceOutputType(parentOnWeights, ov::element::f32).get(),
                                ov::Strides{1, 1},
                                ov::CoordinateDiff{0, 0},
                                ov::CoordinateDiff{0, 0},
                                ov::Strides{1, 1});

    const std::shared_ptr<ov::opset1::Convolution> convolution =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            convolutionOriginal,
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{});
    convolution->set_friendly_name("convolution");

    for (const auto& r : restrictions) {
        for (const auto& restrictedPort : r.restrictions) {
            auto& rt = convolution->input(restrictedPort.port).get_rt_info();
            rt[ov::QuantizationGranularityAttribute::get_type_info_static()] = ov::QuantizationGranularityAttribute(restrictedPort.granularity);
        }
    }

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolution) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ConvolutionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
