// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/fake_quantize_and_two_output_branches_with_convolution.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "low_precision/network_helper.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::opset1::Convolution> createConvolution(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const std::shared_ptr<Node>& parent,
    const FakeQuantizeOnWeights& fqOnWeights,
    bool typeRelaxed) {
    const size_t inputChannelsCount = inputShape[1].get_length();
    const size_t outputChannelsCount = 2 * inputShape[1].get_length();
    const auto weights = ov::opset1::Constant::create(
        precision,
        ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, 1));

    const std::shared_ptr<ov::opset1::Convolution> convolution =
        typeRelaxed ? std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
                          std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
                          std::vector<ov::element::Type>{},
                          ov::op::TemporaryReplaceOutputType(parent, ov::element::f32).get(),
                          ov::op::TemporaryReplaceOutputType(
                              fqOnWeights.empty() ? weights
                                                  : ov::test::utils::make_fake_quantize(weights,
                                                                                        precision,
                                                                                        fqOnWeights.quantizationLevel,
                                                                                        fqOnWeights.constantShape,
                                                                                        fqOnWeights.inputLowValues,
                                                                                        fqOnWeights.inputHighValues,
                                                                                        fqOnWeights.outputLowValues,
                                                                                        fqOnWeights.outputHighValues),
                              ov::element::f32)
                              .get(),
                          ov::Strides{1, 1},
                          ov::CoordinateDiff{0, 0},
                          ov::CoordinateDiff{0, 0},
                          ov::Strides{1, 1})
                    : std::make_shared<ov::opset1::Convolution>(
                          parent,
                          fqOnWeights.empty() ? weights->output(0)
                                              : ov::test::utils::make_fake_quantize(weights,
                                                                                    precision,
                                                                                    fqOnWeights.quantizationLevel,
                                                                                    fqOnWeights.constantShape,
                                                                                    fqOnWeights.inputLowValues,
                                                                                    fqOnWeights.inputHighValues,
                                                                                    fqOnWeights.outputLowValues,
                                                                                    fqOnWeights.outputHighValues),
                          ov::Strides{1, 1},
                          ov::CoordinateDiff{0, 0},
                          ov::CoordinateDiff{0, 0},
                          ov::Strides{1, 1});

    return convolution;
}

std::shared_ptr<ov::Model> FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData,
    const FakeQuantizeOnWeights fqOnWeights1,
    FakeQuantizeOnWeights fqOnWeights2) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = fqOnData.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            input,
            precision,
            fqOnData.quantizationLevel,
            fqOnData.constantShape,
            fqOnData.inputLowValues,
            fqOnData.inputHighValues,
            fqOnData.outputLowValues,
            fqOnData.outputHighValues);

    const std::shared_ptr<ov::opset1::Convolution> convolution1 = createConvolution(
        precision,
        inputShape,
        fakeQuantizeOnActivations,
        fqOnWeights1,
        false);

    const std::shared_ptr<ov::opset1::Convolution> convolution2 = createConvolution(
        precision,
        inputShape,
        fakeQuantizeOnActivations,
        fqOnWeights2,
        false);

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(NodeVector{ convolution1, convolution2 }, 1ul);
    ov::ResultVector results { std::make_shared<ov::opset1::Result>(concat) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction");
}

std::shared_ptr<ov::Model> FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getReference(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const ov::pass::low_precision::LayerTransformation::Params& params,
    const ov::builder::subgraph::FakeQuantizeOnData& fqOnData,
    const ov::element::Type precisionBeforeOp,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOp,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter1,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter2) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    const auto fakeQuantizeOnActivations = fqOnData.empty() ?
        nullptr :
        makeFakeQuantizeTypeRelaxed(input, precision, fqOnData);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantizeOnActivations, precisionBeforeOp);
    const auto deqBefore = makeDequantization(fakeQuantizeOnActivations, dequantizationBefore);

    const std::shared_ptr<ov::opset1::Convolution> convolution1 = createConvolution(
        precision,
        inputShape,
        deqBefore,
        FakeQuantizeOnWeights(),
        true);
    const auto deqAfter1 = makeDequantization(convolution1, dequantizationAfter1);

    const std::shared_ptr<ov::opset1::Convolution> convolution2 = createConvolution(
        precision,
        inputShape,
        deqBefore,
        FakeQuantizeOnWeights(),
        true);
    const auto deqAfter2 = makeDequantization(convolution2, dequantizationAfter2);

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(NodeVector{ deqAfter1, deqAfter2 }, 1ul);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOp);
    if (params.updatePrecisions) {
        replace_node(convolution1->get_input_node_shared_ptr(1),
                     ov::pass::low_precision::fold<ov::opset1::Convert>(convolution1->get_input_node_shared_ptr(1),
                                                                        ov::element::i8));

        replace_node(convolution2->get_input_node_shared_ptr(1),
                     ov::pass::low_precision::fold<ov::opset1::Convert>(convolution2->get_input_node_shared_ptr(1),
                                                                        ov::element::i8));
    }

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(concat) };
    auto function = std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
