// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/group_convolution.hpp"

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::opset1;
using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<Node> createWeightsOriginal(
    const long long int rankLength,
    const ov::element::Type precision,
    const size_t inputChannelsCount,
    const size_t outputChannelsCount,
    const size_t groupCount,
    const int calculatedDimention,
    const size_t kernelSize,
    const std::vector<float>& weightsValues,
    const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
    const ov::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const bool addReshape = true) {
    std::shared_ptr<Node> weights;
    if (fakeQuantizeOnWeights.empty() && dequantizationOnWeights.empty()) {
        weights = ov::opset1::Constant::create(
            precision,
            rankLength == 3 ?
                ov::Shape{ outputChannelsCount, inputChannelsCount, 1 } :
                ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
            weightsValues.size() == 1ul ?
                std::vector<float>(outputChannelsCount * inputChannelsCount, weightsValues[0]) :
                weightsValues);
    } else {
        const size_t inputChannelsPerGroup = inputChannelsCount / groupCount;
        if ((rankLength == 3) || (rankLength == 4)) {
            weights = ov::opset1::Constant::create(
                precision,
                addReshape ?
                    (rankLength == 3 ?
                        ov::Shape{ outputChannelsCount, inputChannelsPerGroup, kernelSize } :
                        ov::Shape{ outputChannelsCount, inputChannelsPerGroup, kernelSize, kernelSize }) :
                    (rankLength == 3 ?
                        ov::Shape{ groupCount, outputChannelsCount / groupCount, inputChannelsPerGroup, kernelSize } :
                        ov::Shape{ groupCount, outputChannelsCount / groupCount, inputChannelsPerGroup, kernelSize, kernelSize }),
                weightsValues.size() == 1ul ?
                    std::vector<float>(
                        rankLength == 3 ?
                            outputChannelsCount * kernelSize * inputChannelsPerGroup :
                            outputChannelsCount * kernelSize * kernelSize * inputChannelsPerGroup,
                        weightsValues[0]) :
                    weightsValues);
        } else {
            const ov::Shape shape{outputChannelsCount, inputChannelsPerGroup, 1ull, kernelSize, kernelSize};
            const std::vector<float> values = weightsValues.size() == 1ull ?
                std::vector<float>(shape_size(shape), weightsValues[0]) :
                weightsValues;
            weights = ov::opset1::Constant::create(precision, shape, values);
        }

        if (!fakeQuantizeOnWeights.empty()) {
            Shape constantShape;
            if (ov::shape_size(fakeQuantizeOnWeights.constantShape) != 1) {
                constantShape = fakeQuantizeOnWeights.constantShape;
            } else {
                constantShape = Shape(weights->get_shape().size(), 1);
                if (addReshape) {
                    constantShape[0] = outputChannelsCount;
                } else {
                    constantShape[0] = groupCount;
                    constantShape[1] = outputChannelsCount / groupCount;
                }
            }
            weights = ov::test::utils::make_fake_quantize(
                weights,
                precision,
                fakeQuantizeOnWeights.quantizationLevel,
                constantShape,
                fakeQuantizeOnWeights.inputLowValues,
                fakeQuantizeOnWeights.inputHighValues,
                fakeQuantizeOnWeights.outputLowValues,
                fakeQuantizeOnWeights.outputHighValues);
        }

        if (!dequantizationOnWeights.empty()) {
            weights = ov::builder::subgraph::makeDequantization(weights, dequantizationOnWeights);
        }

        if (addReshape) {
            std::vector<int64_t> values;
            if (rankLength == 3ll) {
                values = std::vector<int64_t>{
                    calculatedDimention == 0 ? -1 : static_cast<int64_t>(groupCount),
                    calculatedDimention == 1 ? -1 : static_cast<int64_t>(outputChannelsCount / groupCount),
                    static_cast<int64_t>(inputChannelsPerGroup),
                    static_cast<int64_t>(kernelSize)};
            } else if (rankLength == 4ll) {
                values = std::vector<int64_t>{
                    calculatedDimention == 0 ? -1 : static_cast<int64_t>(groupCount),
                    calculatedDimention == 1 ? -1 : static_cast<int64_t>(outputChannelsCount / groupCount),
                    static_cast<int64_t>(inputChannelsPerGroup),
                    static_cast<int64_t>(kernelSize),
                    static_cast<int64_t>(kernelSize)};
            } else if (rankLength == 5ll) {
                values = std::vector<int64_t>{
                    calculatedDimention == 0 ? -1 : static_cast<int64_t>(groupCount),
                    calculatedDimention == 1 ? -1 : static_cast<int64_t>(outputChannelsCount / groupCount),
                    static_cast<int64_t>(inputChannelsPerGroup),
                    1,
                    static_cast<int64_t>(kernelSize),
                    static_cast<int64_t>(kernelSize)};
            }

            weights = std::make_shared<ov::opset1::Reshape>(
                weights,
                ov::opset1::Constant::create(ov::element::i64, Shape{static_cast<size_t>(rankLength) + 1ul}, values),
                true);
        }
    }

    return weights;
}

std::shared_ptr<ov::Model> GroupConvolutionFunction::getOriginal(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const ov::Shape& outputShape,
    const size_t groupCount,
    const int groupCalculationDimention,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ov::opset1::Constant> weightsConst,
    const ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights) {
    const auto rankLength = inputShape.size();
    OPENVINO_ASSERT(rankLength == 3 || rankLength == 4, "not supported input shape rank: ", rankLength);

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto dequantization = makeDequantization(input, dequantizationBefore);

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 7ul;

    const size_t weightsSize = weightsConst->cast_vector<float>().size();
    if ((weightsSize != 1ul) && (weightsSize != (inputChannelsCount * outputChannelsCount))) {
        throw std::runtime_error("unexpected actual weights values size");
    }

    std::shared_ptr<ov::Node> weights = createWeightsOriginal(
        rankLength,
        weightsConst->get_element_type(),
        inputChannelsCount,
        outputChannelsCount,
        groupCount,
        groupCalculationDimention,
        kernelSize,
        weightsConst->cast_vector<float>(),
        fakeQuantizeOnWeights,
        {});

    const auto convolution = std::make_shared<ov::opset1::GroupConvolution>(
        dequantization,
        weights,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    convolution->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolution) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "GroupConvolutionTransformation");
}

std::shared_ptr<ov::Model> GroupConvolutionFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::Shape& outputShape,
    const size_t groupCount,
    const int groupCalculationDimention,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const FakeQuantizeOnWeights& fakeQuantizeOnWeights,
    const bool addReshape,
    const bool addPrecisionPreserved) {
    const auto rankLength = inputShape.rank().is_dynamic() ? 4 : inputShape.rank().get_length();
    OPENVINO_ASSERT(rankLength == 3 || rankLength == 4 || rankLength == 5, "not supported input shape rank: ", rankLength);

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);

    std::shared_ptr<ov::Node> parent = input;
    if (!fakeQuantizeOnData.empty()) {
        parent = std::make_shared<ov::opset1::FakeQuantize>(
            input,
            std::make_shared<ov::opset1::Constant>(
                precision,
                rankLength == 3 ?
                    Shape{ 1, fakeQuantizeOnData.inputLowValues.size(), 1 } :
                    Shape{ 1, fakeQuantizeOnData.inputLowValues.size(), 1, 1 },
                fakeQuantizeOnData.inputLowValues),
            std::make_shared<ov::opset1::Constant>(
                precision,
                rankLength == 3 ?
                    Shape{ 1, fakeQuantizeOnData.inputHighValues.size(), 1 } :
                    Shape{ 1, fakeQuantizeOnData.inputHighValues.size(), 1, 1 },
                fakeQuantizeOnData.inputHighValues),
            std::make_shared<ov::opset1::Constant>(
                precision,
                rankLength == 3 ?
                    Shape{ 1, fakeQuantizeOnData.outputLowValues.size(), 1 } :
                    Shape{ 1, fakeQuantizeOnData.outputLowValues.size(), 1, 1 },
                fakeQuantizeOnData.outputLowValues),
            std::make_shared<ov::opset1::Constant>(
                precision,
                rankLength == 3 ?
                    Shape{ 1, fakeQuantizeOnData.outputHighValues.size(), 1 } :
                    Shape{ 1, fakeQuantizeOnData.outputHighValues.size(), 1, 1 },
                fakeQuantizeOnData.outputHighValues),
            fakeQuantizeOnData.quantizationLevel);
    }

    if (addPrecisionPreserved) {
        const std::vector<size_t> stride(rankLength - 2, 1ul);
        const std::vector<size_t> padBegin(rankLength - 2, 0ul);
        const std::vector<size_t> padEnd(rankLength - 2, 0ul);
        const std::vector<size_t> padKernel(rankLength - 2, 3ul);
        const ov::op::PadType padType = ov::op::PadType::NOTSET;
        const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;
        const auto pooling = std::make_shared<ov::opset1::MaxPool>(
            parent,
            stride,
            padBegin,
            padEnd,
            padKernel,
            roundingType,
            padType);
        parent = pooling;
    }

    const size_t outputChannelsCount = outputShape[1];
    const size_t kernelSize = 5ul;
    const size_t inputChannelsCount = inputShape[1].get_length();

    std::vector<float> weightsValues = { 1.f };
    std::shared_ptr<ov::Node> weights = createWeightsOriginal(
        rankLength,
        precision,
        inputChannelsCount,
        outputChannelsCount,
        groupCount,
        groupCalculationDimention,
        kernelSize,
        weightsValues,
        fakeQuantizeOnWeights,
        {},
        addReshape);

    const auto convolution = std::make_shared<ov::opset1::GroupConvolution>(
        parent,
        weights,
        ov::Strides(rankLength - 2, 1ul),
        ov::CoordinateDiff(rankLength - 2, 0),
        ov::CoordinateDiff(rankLength - 2, 0),
        ov::Strides(rankLength - 2, 1));

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolution) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "GroupConvolutionTransformation");
}

std::shared_ptr<ov::Model> GroupConvolutionFunction::get(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::PartialShape& outputShape,
    const size_t groupCount,
    const int calculatedDimention,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    std::shared_ptr<ov::opset1::Constant> weightsConst,
    const ov::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
    const ov::builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const ov::element::Type precisionAfterDequantization,
    const bool addReshape) {
    const auto rankLength = inputShape.rank().is_dynamic() ? 4 : inputShape.rank().get_length();
    OPENVINO_ASSERT(rankLength == 3 || rankLength == 4, "not supported input shape rank: ", rankLength);

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const bool channelsIsDynamic = inputShape.rank().is_dynamic() || inputShape[1].is_dynamic();
    const size_t inputChannelsCount = !channelsIsDynamic ? inputShape[1].get_length() : 6ul;

    const size_t outputChannelsCount = !channelsIsDynamic ? outputShape[1].get_length() : 24ul;
    const size_t kernelSize = 7ul;
    const size_t inputChannelsInGroup = inputChannelsCount / groupCount;
    const size_t outputChannelsInGroup = outputChannelsCount / groupCount;

    const size_t weightsSize = weightsConst->cast_vector<float>().size();

    std::shared_ptr<ov::Node> weights;
    if (fakeQuantizeOnWeights.empty() && dequantizationOnWeights.empty()) {
        const ov::Shape weightsShape = ov::Shape{ groupCount, outputChannelsInGroup, inputChannelsInGroup, kernelSize, kernelSize };
        weights = ov::opset1::Constant::create(
            weightsConst->get_element_type(),
            weightsShape,
            weightsSize == 1ul ? std::vector<float>(
                groupCount * outputChannelsInGroup * inputChannelsInGroup * kernelSize * kernelSize,
                weightsConst->cast_vector<float>()[0]) : weightsConst->cast_vector<float>());
    } else {
        weights = createWeightsOriginal(
            rankLength,
            weightsConst->get_element_type(),
            inputChannelsCount,
            outputChannelsCount,
            groupCount,
            calculatedDimention,
            kernelSize,
            weightsConst->cast_vector<float>(),
            fakeQuantizeOnWeights,
            dequantizationOnWeights,
            addReshape);
    }

    auto convolutionOriginal =
        ov::opset1::GroupConvolution(ov::op::TemporaryReplaceOutputType(deqBefore, ov::element::f32).get(),
                                     ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
                                     ov::Strides{1, 1},
                                     ov::CoordinateDiff{0, 0},
                                     ov::CoordinateDiff{0, 0},
                                     ov::Strides{1, 1});

    std::shared_ptr<ov::opset1::GroupConvolution> convolution =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::GroupConvolution>>(
            convolutionOriginal,
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{});
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(convolution, precisionAfterOperation);

    const auto deqAfter = makeDequantization(convolution, dequantizationAfter);
    deqAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(deqAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "GroupConvolutionTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
