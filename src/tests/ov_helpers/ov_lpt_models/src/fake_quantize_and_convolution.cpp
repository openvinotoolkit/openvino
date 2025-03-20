// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fake_quantize_and_convolution.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

// TODO: remove, reuse mode extended method
std::shared_ptr<ov::Model> FakeQuantizeAndConvolutionFunction::get(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const FakeQuantizeOnWeights& fqOnWeights) {
    const auto rankLength = inputShape.rank().is_dynamic() ? 4 : inputShape.rank().get_length();
    OPENVINO_ASSERT(rankLength == 3ul || rankLength == 4ul || rankLength == 5ul, "not supported input shape rank: ", rankLength);

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = fqOnData.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            input, precision, fqOnData.quantizationLevel, fqOnData.constantShape,
            fqOnData.inputLowValues, fqOnData.inputHighValues, fqOnData.outputLowValues, fqOnData.outputHighValues);
    if (fakeQuantizeOnActivations != nullptr) {
        fakeQuantizeOnActivations->set_friendly_name("fakeQuantizeOnActivations");
    }

    const size_t inputChannelsCount = inputShape[1].get_length();
    const size_t outputChannelsCount = 2 * inputShape[1].get_length();
    const auto weights = ov::opset1::Constant::create(
        precision,
        rankLength == 3ul ?
            (ov::Shape{ outputChannelsCount, inputChannelsCount, 1}) :
            (ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 }),
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    auto maxPool = std::make_shared<ov::opset1::MaxPool>(fqOnData.empty() ? input : fakeQuantizeOnActivations,
                                                         Strides(rankLength - 2, 1ul),
                                                         Shape(rankLength - 2, 1ul),
                                                         Shape(rankLength - 2, 0ul),
                                                         Shape(rankLength - 2, 2ul),
                                                         ov::op::RoundingType::FLOOR);
    maxPool->set_friendly_name("maxPool");

    const auto convolution = std::make_shared<ov::opset1::Convolution>(
        maxPool, //fqOnData.empty() ? input : fakeQuantizeOnActivations,
        fqOnWeights.empty() ?
            weights->output(0) :
            ov::test::utils::make_fake_quantize(
                weights, precision, fqOnWeights.quantizationLevel, fqOnWeights.constantShape,
                fqOnWeights.inputLowValues, fqOnWeights.inputHighValues, fqOnWeights.outputLowValues, fqOnWeights.outputHighValues),
        ov::Strides(rankLength - 2, 1ul),
        ov::CoordinateDiff(rankLength - 2, 0ul),
        ov::CoordinateDiff(rankLength - 2, 0ul),
        ov::Strides(rankLength - 2, 1ul));
    convolution->set_friendly_name("convolution");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolution) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizeAndConvolutionFunction");
}

std::shared_ptr<ov::Model> FakeQuantizeAndConvolutionFunction::get(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
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

std::shared_ptr<ov::Model> FakeQuantizeAndConvolutionFunction::get(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
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
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);

    std::shared_ptr<Node> parentOnActivation = input;
    {
        if (!fqOnData.empty()) {
            parentOnActivation = fqOnData.outputPrecision == element::dynamic
                                     ? ov::builder::subgraph::makeFakeQuantize(input, precision, fqOnData)
                                     : ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(input, precision, fqOnData);
        }

        if (!convertOnData.empty()) {
            parentOnActivation = std::make_shared<ov::opset1::Convert>(parentOnActivation, convertOnData.outPrecision);
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

        const Shape shape = constantOnWeights.shapeIsDefined ? constantOnWeights.shape : ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 };
        parentOnWeights = ov::opset1::Constant::create(
            constantOnWeights.outPrecision,
            shape,
            constantOnWeights.values.size() != ov::shape_size(shape) ?
                std::vector<float>(ov::shape_size(shape), constantOnWeights.values[0]) :
                constantOnWeights.values);

        if (!fqOnWeights.empty()) {
            parentOnWeights =
                fqOnWeights.outputPrecision == element::dynamic
                    ? ov::builder::subgraph::makeFakeQuantize(parentOnWeights,
                                                              parentOnWeights->output(0).get_element_type(),
                                                              fqOnWeights)
                    : ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(parentOnWeights,
                                                                         parentOnWeights->output(0).get_element_type(),
                                                                         fqOnWeights);
        }

        if (!convertOnWeights.empty()) {
            parentOnWeights = std::make_shared<ov::opset1::Convert>(parentOnWeights, convertOnWeights.outPrecision);
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
        lastOperation = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            ov::opset1::Convolution(ov::op::TemporaryReplaceOutputType(parentOnActivation, ov::element::f32).get(),
                                    ov::op::TemporaryReplaceOutputType(parentOnWeights, ov::element::f32).get(),
                                    ov::Strides{1, 1},
                                    ov::CoordinateDiff{0, 0},
                                    ov::CoordinateDiff{0, 0},
                                    ov::Strides{1, 1}),
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{});
    } else if (operation == "GroupConvolution") {
        lastOperation = std::make_shared<ov::op::TypeRelaxed<ov::opset1::GroupConvolution>>(
            ov::opset1::GroupConvolution(ov::op::TemporaryReplaceOutputType(parentOnActivation, ov::element::f32).get(),
                                         ov::op::TemporaryReplaceOutputType(parentOnWeights, ov::element::f32).get(),
                                         ov::Strides{1, 1},
                                         ov::CoordinateDiff{0, 0},
                                         ov::CoordinateDiff{0, 0},
                                         ov::Strides{1, 1}),
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{});
        if (multiplyAfter) {
            const auto& O = lastOperation->get_shape()[1];
            std::vector<float> weights_val(O, 1);
            auto constant = ov::opset1::Constant::create(ov::element::f32, Shape{O, 1, 1}, weights_val);
            lastOperation = std::make_shared<ov::opset1::Multiply>(lastOperation, constant);
        }
    } else {
        OPENVINO_THROW("Unknown operation type ", operation);
    }

    if (!dequantizationAfter.empty()) {
        lastOperation->set_friendly_name("output_original");
        lastOperation = makeDequantization(lastOperation, dequantizationAfter);
        lastOperation->set_friendly_name("output");
    } else {
        lastOperation->set_friendly_name("output");
    }

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastOperation) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizeAndConvolutionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
