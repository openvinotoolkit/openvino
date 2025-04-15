// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fake_quantize_precision_selection.hpp"

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "ov_lpt_models/common/builders.hpp"
#include "low_precision/network_helper.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> FakeQuantizePrecisionSelectionFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ActualValues& values) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");
    std::shared_ptr<ov::Node> parent = input;

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
        input, precision,
        values.fakeQuantizeOnData.quantizationLevel,
        values.fakeQuantizeOnData.constantShape,
        values.fakeQuantizeOnData.inputLowValues,
        values.fakeQuantizeOnData.inputHighValues,
        values.fakeQuantizeOnData.outputLowValues,
        values.fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    std::shared_ptr<ov::Node> branch1Last;
    {
        // branch with limitation precision operation (Convolution)
        std::shared_ptr<ov::Node> branch1Operation =
            values.operationBeforeLimitedOperationIsPrecisionTransparent
                ? std::dynamic_pointer_cast<ov::Node>(
                      std::make_shared<ov::opset1::MaxPool>(fakeQuantize,
                                                            Strides{1, 1},
                                                            Shape{1, 1},
                                                            Shape{0, 0},
                                                            Shape{2, 2},
                                                            op::RoundingType::FLOOR))
                : std::make_shared<ov::op::TypeRelaxed<ov::opset1::PRelu>>(
                      ov::opset1::PRelu(
                          fakeQuantize,
                          std::make_shared<ov::opset1::Constant>(element::f32, Shape{}, std::vector<float>{0.01})),
                      element::f32);

        const size_t inputChannelsCount = inputShape[1].get_length();
        const size_t outputChannelsCount = 2 * inputShape[1].get_length();

        const auto weights = ov::opset1::Constant::create(
            precision,
            ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
            std::vector<float>(outputChannelsCount * inputChannelsCount, 1.f));

        std::shared_ptr<ov::opset1::Convolution> convolution = std::make_shared<ov::opset1::Convolution>(
            branch1Operation,
            values.fakeQuantizeOnWeights.empty() ?
                weights->output(0) :
                ov::test::utils::make_fake_quantize(
                    weights,
                    precision,
                    values.fakeQuantizeOnWeights.quantizationLevel,
                    values.fakeQuantizeOnWeights.constantShape,
                    values.fakeQuantizeOnWeights.inputLowValues,
                    values.fakeQuantizeOnWeights.inputHighValues,
                    values.fakeQuantizeOnWeights.outputLowValues,
                    values.fakeQuantizeOnWeights.outputHighValues),
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });

        branch1Last = convolution;
    }

    std::shared_ptr<ov::Node> branch2Last;
    {
        // just another branch
        branch2Last = std::make_shared<ov::op::TypeRelaxed<ov::opset1::PRelu>>(
            ov::opset1::PRelu(
                fakeQuantize,
                std::make_shared<ov::opset1::Constant>(ov::element::f32, Shape{}, std::vector<float>{0.01})),
            ov::element::f32);
    }

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ branch1Last->output(0), branch2Last->output(0) }, 1);
    concat->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(concat) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizePrecisionSelectionFunction");
}

std::shared_ptr<ov::Model> FakeQuantizePrecisionSelectionFunction::getReference(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const ExpectedValues& values) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(
        input,
        precision,
        values.fakeQuantizeOnData);
    fakeQuantize->set_friendly_name("fakeQuantize");

    // branch with limitation precision operation (Convolution)
    std::shared_ptr<ov::Node> branch1Pooling =
        values.operationBeforeLimitedOperationIsPrecisionTransparent
            ? std::dynamic_pointer_cast<ov::Node>(std::make_shared<ov::opset1::MaxPool>(fakeQuantize,
                                                                                        Strides{1, 1},
                                                                                        Shape{1, 1},
                                                                                        Shape{0, 0},
                                                                                        Shape{2, 2},
                                                                                        op::RoundingType::FLOOR))
            : std::make_shared<ov::op::TypeRelaxed<ov::opset1::PRelu>>(
                  fakeQuantize,
                  std::make_shared<ov::opset1::Constant>(element::f32, Shape{}, std::vector<float>{0.01}));

    const size_t inputChannelsCount = inputShape[1];
    const size_t outputChannelsCount = 2 * inputShape[1];

    const auto weights = ov::opset1::Constant::create(
        precision,
        ov::Shape{ outputChannelsCount, inputChannelsCount, 1, 1 },
        std::vector<float>(outputChannelsCount * inputChannelsCount, -126.f));

    std::shared_ptr<ov::Node> onWeights = values.fakeQuantizeOnWeights.empty() ?
        weights :
        ov::test::utils::make_fake_quantize(
            weights,
            precision,
            values.fakeQuantizeOnWeights.quantizationLevel,
            values.fakeQuantizeOnWeights.constantShape,
            values.fakeQuantizeOnWeights.inputLowValues,
            values.fakeQuantizeOnWeights.inputHighValues,
            values.fakeQuantizeOnWeights.outputLowValues,
            values.fakeQuantizeOnWeights.outputHighValues);

    std::shared_ptr<ov::opset1::Convolution> convolution =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{},
            ov::op::TemporaryReplaceOutputType(branch1Pooling, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(onWeights, ov::element::f32).get(),
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});

    std::shared_ptr<ov::opset1::Multiply> branch1Multiply = std::make_shared<ov::opset1::Multiply>(
        convolution,
        std::make_shared<ov::opset1::Constant>(precision, Shape({}), std::vector<float>({ 0.0001f })));


    // just another branch
    std::shared_ptr<ov::opset1::PRelu> branch2PRelu = std::make_shared<ov::op::TypeRelaxed<ov::opset1::PRelu>>(
        fakeQuantize,
        std::make_shared<ov::opset1::Constant>(ov::element::f32, Shape{}, std::vector<float>{0.01}));

    const std::shared_ptr<ov::Node> branch2Multiply = std::make_shared<ov::opset1::Multiply>(
        branch2PRelu,
        std::make_shared<ov::opset1::Constant>(precision, Shape({}), std::vector<float>({0.01f})));

    if (values.fakeQuantizeOnDataOutPrecision != precision) {
        ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize, values.fakeQuantizeOnDataOutPrecision);

        if (values.operationBeforeLimitedOperationIsPrecisionTransparent) {
            auto intermediateOpTr = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(branch1Pooling);
            if (intermediateOpTr != nullptr) {
                ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(branch1Pooling, values.fakeQuantizeOnDataOutPrecision);
            } else {
                // TODO: potential workaround for the same case:
                // openvino\inference-engine\tests\ov_models\src\low_precision_transformations\concat_function.cpp, line #496
                 branch1Pooling->set_output_type(0, values.fakeQuantizeOnDataOutPrecision, branch1Pooling->get_output_partial_shape(0));
            }
        }

        if (values.fakeQuantizeOnWeights.empty()) {
            replace_node(
                weights,
                ov::pass::low_precision::fold<ov::opset1::Convert>(weights, ov::element::i8));
        }

        ov::pass::low_precision::NetworkHelper::setOutDataPrecision(branch2PRelu, precision);
    }


    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ branch1Multiply->output(0), branch2Multiply->output(0) }, 1);
    concat->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(concat) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FakeQuantizePrecisionSelectionFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
