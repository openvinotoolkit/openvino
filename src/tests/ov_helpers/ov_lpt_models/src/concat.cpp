// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/concat.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> ConcatFunction::get(
    const ov::element::Type inputPrecision,
    const ov::element::Type deqPrecision,
    const std::vector<ov::PartialShape>& inputShapes,
    const std::vector<DequantizationOperations>& dequantizationsBefore,
    const std::int64_t concatAxis,
    const ov::element::Type precisionAfter,
    const DequantizationOperations& dequantizationAfter) {
    auto modifyDeq = [](const DequantizationOperations& deq, const ov::element::Type deqOutPrc) {
        auto dequantizationStructure = deq;
        if (!dequantizationStructure.multiply.empty()) {
            dequantizationStructure.multiply.outPrecision = deqOutPrc;
        }
        return dequantizationStructure;
    };

    ov::ParameterVector inputs;
    ov::NodeVector concatInputs;
    if (inputShapes.size() != dequantizationsBefore.size()) {
        throw std::runtime_error("Concat builder: input and dequantization sizes aren't equal");
    }

    for (size_t i = 0; i < inputShapes.size(); ++i) {
        const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShapes[i]);
        const auto dequantization = makeDequantization(input, modifyDeq(dequantizationsBefore[i], deqPrecision));
        inputs.push_back(input);
        concatInputs.push_back(dequantization);
    }

    const auto concat = std::make_shared<ov::opset1::Concat>(concatInputs, concatAxis);
    if (precisionAfter != ov::element::dynamic &&
        (concat->get_output_element_type(0).is_real() ^ precisionAfter.is_real())) {
        throw std::runtime_error("Concat builder: requested precision after operation could't be set");
    }

    const auto deqAfter = makeDequantization(concat, modifyDeq(dequantizationAfter, deqPrecision));
    deqAfter->set_friendly_name("Concat");
    const auto result = std::make_shared<ov::opset1::Result>(deqAfter);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, inputs, "ConcatTransformation");
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const std::shared_ptr<ov::opset1::Constant>& input_constant1,
    const FakeQuantizeOnData& fqOnData1,
    const DequantizationOperations& dequantization1,
    const std::shared_ptr<ov::opset1::Constant>& input_constant2,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations& dequantization2) {
    std::shared_ptr<Node> parent1;
    std::shared_ptr<ov::opset1::Parameter> input1;
    if (input_constant1 == nullptr) {
        input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
        input1->set_friendly_name("input1");
        parent1 = input1;
    } else {
        parent1 = input_constant1;
    }

    if (!fqOnData1.empty()) {
        parent1 = makeFakeQuantize(parent1, precision, fqOnData1);
    }
    if (!dequantization1.empty()) {
        parent1 = makeDequantization(parent1, dequantization1);
    }

    std::shared_ptr<Node> parent2;
    std::shared_ptr<ov::opset1::Parameter> input2;
    if (input_constant2 == nullptr) {
        input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
        input2->set_friendly_name("input2");
        parent2 = input2;
    } else {
        parent2 = input_constant2;
    }

    if (!fqOnData2.empty()) {
        parent2 = makeFakeQuantize(parent2, precision, fqOnData2);
    }
    if (!dequantization2.empty()) {
        parent2 = makeDequantization(parent2, dequantization2);
    }

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ parent1->output(0), parent2->output(0) }, 1);
    concat->set_friendly_name("output");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    ov::ParameterVector inputs;
    if (input1 != nullptr) {
        inputs.push_back(input1);
    }
    if (input2 != nullptr) {
        inputs.push_back(input2);
    }

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(concat) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        inputs,
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginal(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat->set_friendly_name("output");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(concat) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithChildAndOutput(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    std::shared_ptr<ov::opset1::Result> res1;
    const auto inputShape2 = inputShape;
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape2);
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const auto concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat->set_friendly_name("110");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const auto clamp = std::make_shared<ov::opset1::Clamp>(concat, 0.0, 6.0);
    clamp->set_friendly_name("111");

    ResultVector results{ std::make_shared<ov::opset1::Result>(clamp), std::make_shared<ov::opset1::Result>(concat) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithChildAndOutputTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithNeighbors(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const std::string& neighborType,
    const std::string& additionalLayer) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto concat1 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { fakeQuantize1->output(0), fakeQuantize2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = "concat1";

    ov::ParameterVector inputs{input1, input2};

    ov::ResultVector results { };
    if (additionalLayer == "convolution") {
        ov::Shape convShape(4);
        convShape[1] = inputShape[1].get_length() + inputShape[1].get_length();
        convShape[0] = convShape[1] * 2;
        convShape[2] = convShape[3] = 1;
        auto convolutionAddition = std::make_shared<ov::opset1::Convolution>(
            concat1,
            std::make_shared<ov::opset1::Multiply>(
                std::make_shared<ov::opset1::Convert>(ov::opset1::Constant::create(ov::element::i8, convShape, {1}),
                                                      ov::element::f32),
                ov::opset1::Constant::create(ov::element::f32, Shape{}, {1})),
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
        convolutionAddition->set_friendly_name("convolution_addition");
        results.push_back(std::make_shared<ov::opset1::Result>(convolutionAddition));
    }
    if (neighborType == "concat") {
        const auto input3 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
        input3->set_friendly_name("input3");
        const auto fakeQuantize3 = makeFakeQuantize(input3, precision, fqOnData3);
        fakeQuantize3->set_friendly_name("fakeQuantize3");
        inputs.push_back(input3);

        const auto concat2 = std::make_shared<ov::opset1::Concat>(
                ov::OutputVector { fakeQuantize2->output(0), fakeQuantize3->output(0) },
                1ull);
        concat2->set_friendly_name("concat2");
        auto& rtInfo2 = concat2->get_rt_info();
        rtInfo2["Variant::std::string"] = "concat2";
        results.push_back(std::make_shared<ov::opset1::Result>(concat1));
        results.push_back(std::make_shared<ov::opset1::Result>(concat2));
    } else if (neighborType == "convolution") {
        ov::Shape convShape(4);
        convShape[0] = inputShape[1].get_length() * 2;
        convShape[1] = inputShape[1].get_length();
        convShape[2] = convShape[3] = 1;
        auto convolutionNeighbor = std::make_shared<ov::opset1::Convolution>(
            fakeQuantize2,
            std::make_shared<ov::opset1::Multiply>(
                std::make_shared<ov::opset1::Convert>(ov::opset1::Constant::create(ov::element::i8, convShape, {1}),
                                                      ov::element::f32),
                ov::opset1::Constant::create(ov::element::f32, Shape{}, {1})),
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
        convolutionNeighbor->set_friendly_name("convolution_neighbor");
        results.push_back(std::make_shared<ov::opset1::Result>(convolutionNeighbor));
    }

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        inputs,
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithIntermediate(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const size_t channels = inputShape[1].is_dynamic() ? 3ul : inputShape[1].get_length();
    ov::PartialShape inputShape1 = inputShape;

    if (inputShape1[2].is_static() && transparentIntermediate) {
        inputShape1[2] = inputShape1[2].get_length() - 2;
    }

    if (inputShape1[3].is_static() && transparentIntermediate) {
        inputShape1[3] = inputShape1[3].get_length() - 2;
    }

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    } else {
        auto weights = ov::opset1::Constant::create(
            precision,
            ov::Shape{ channels, channels, 1, 1 },
            std::vector<float>(channels * channels, 1));

        intermediateOp = std::make_shared<ov::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    auto weights = ov::opset1::Constant::create(precision, ov::Shape{ channels, channels, 1, 1 }, { 1 });
    auto convolution = std::make_shared<ov::opset1::Convolution>(
        intermediateOp,
        weights,
        ov::Strides { 1, 1 },
        ov::CoordinateDiff { 0, 0 },
        ov::CoordinateDiff { 0, 0 },
        ov::Strides { 1, 1 });
    convolution->set_friendly_name("convolution");

    ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(concat),
        std::make_shared<ov::opset1::Result>(convolution)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithIntermediateAvgPool(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    ov::PartialShape inputShape1 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    if (inputShape1[2].is_static()) {
        inputShape1[2] = inputShape[2].get_length() - 2;
    }
    if (inputShape1[3].is_static()) {
        inputShape1[3] = inputShape[3].get_length() - 2;
    }

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const ov::PartialShape inputShape2 = inputShape;
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape2);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    std::shared_ptr<Node> parent2 = std::make_shared<ov::opset1::AvgPool>(intermediateOp,
                                                                          Strides{1, 1},
                                                                          Shape{1, 1},
                                                                          Shape{0, 0},
                                                                          Shape{2, 2},
                                                                          true,
                                                                          ov::op::RoundingType::FLOOR);
    parent2->set_friendly_name("avgPool");

    ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(concat),
        std::make_shared<ov::opset1::Result>(parent2)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithSplitedIntermediate(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const bool addConvolution) {
    size_t numSplit = 2;
    size_t splitedAxis = 1;

    auto inputShape1 = inputShape;
    inputShape1[1] = inputShape1[1].get_length() / numSplit;

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<ov::op::Op> intermediateOp;

    const auto constant = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{}, splitedAxis);
    intermediateOp = std::make_shared<ov::opset1::Split>(fakeQuantize2->output(0), constant, numSplit);

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, splitedAxis);
    concat->set_friendly_name("output_1");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    ov::Output<Node> lastOutput = intermediateOp->output(1);
    if (addConvolution) {
        auto weights = ov::opset1::Constant::create(
            precision, ov::Shape{ static_cast<size_t>(inputShape[1].get_length() / numSplit),
                                      static_cast<size_t>(inputShape[1].get_length() / numSplit), 1, 1 }, { 1 });
        auto convolution = std::make_shared<ov::opset1::Convolution>(
            intermediateOp->output(1),
            weights,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
        lastOutput = convolution->output(0);
    }
    lastOutput.get_node_shared_ptr()->set_friendly_name("output_2");

    ov::ResultVector results{
        std::make_shared<ov::opset1::Result>(concat),
        std::make_shared<ov::opset1::Result>(lastOutput),
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalSelectionWithIntermediate(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    } else {
        auto weights = ov::opset1::Constant::create(
            precision,
            ov::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ov::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    auto weights = ov::opset1::Constant::create(precision, ov::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ov::opset1::Convolution>(
        intermediateOp,
        weights,
        ov::Strides { 1, 1 },
        ov::CoordinateDiff { 0, 0 },
        ov::CoordinateDiff { 0, 0 },
        ov::Strides { 1, 1 });
    convolution->set_friendly_name("convolution");

    ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(concat),
        std::make_shared<ov::opset1::Result>(convolution)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

/*
(SS) - optional

        Input
          /
         FQ
        /  \
      (SS) Clamp
        |    |
        |    FQ
        \    /
        Concat
          /\
         /  \
       (SS) MaxPool
*/

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithStridedSlice(
    const ov::element::Type precision,
    const ov::PartialShape inputShape,
    const FakeQuantizeOnData& fq1,
    const FakeQuantizeOnData& fq2,
    const bool ssBeforeConcat,
    const bool ssAfterConcat) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantize(input, precision, fq1);
    fakeQuantize1->set_friendly_name("FakeQuantize_1");

    std::shared_ptr<ov::Node> parent1 = fakeQuantize1;

    const size_t rank = inputShape.rank().get_length();
    if (ssBeforeConcat) {
        const auto beginParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>(rank, 0));

        const auto endParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>{ 0, inputShape[1].get_length() - 2, 0, 0 });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        parent1 = std::make_shared<ov::opset1::StridedSlice>(parent1, beginParam, endParam, beginMask, endMask);
        parent1->set_friendly_name("StridedSlice_1");
    }

    const auto clamp = std::make_shared<ov::opset1::Clamp>(fakeQuantize1, 0.0, 6.0);
    clamp->set_friendly_name("Clamp");
    const auto fakeQuantize2 = makeFakeQuantize(clamp, precision, fq2);
    fakeQuantize2->set_friendly_name("FakeQuantize_2");

    const auto concat = std::make_shared<ov::opset1::Concat>(NodeVector{ parent1, fakeQuantize2 }, 1);
    concat->set_friendly_name("Concat");


    ov::ResultVector results;
    if (ssAfterConcat) {
        const auto concatShape = concat->get_output_partial_shape(0);
        const auto beginParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>(rank, 0));

        const auto endParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>{ 0, concatShape[1].get_length() - 2, 0, 0 });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        const auto stridedSlice = std::make_shared<ov::opset1::StridedSlice>(concat, beginParam, endParam, beginMask, endMask);
        stridedSlice->set_friendly_name("StridedSlice_2");

        const auto result1 = std::make_shared<ov::opset1::Result>(stridedSlice);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    } else {
        const auto result1 = std::make_shared<ov::opset1::Result>(concat);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    }

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

    const auto maxPool = std::make_shared<ov::opset1::MaxPool>(
        concat,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    maxPool->set_friendly_name("MaxPool");

    const std::shared_ptr<Node> convolution = makeConvolution(maxPool, precision, false);

    const auto result2 = std::make_shared<ov::opset1::Result>(convolution);
    result2->set_friendly_name("Result_2");
    results.push_back(result2);

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithDifferentPrecisionOnChildren(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const std::int64_t axis,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const auto concat = std::make_shared<ov::opset1::Concat>(OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, axis);

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ov::opset1::AvgPool>(
        concat->output(0),
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("AvgPool");

    const auto maxPool = std::make_shared<ov::opset1::MaxPool>(
        concat->output(0),
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    maxPool->set_friendly_name("MaxPool");

    ov::ResultVector results;
    results.push_back(std::make_shared<ov::opset1::Result>(avgPool));
    results.push_back(std::make_shared<ov::opset1::Result>(maxPool));

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithIntermediateWithConstant(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");


    std::shared_ptr<ov::op::Op> intermediateOp;

    if (transparentIntermediate) {
        const auto pooling = makeMaxPool(fakeQuantize1->output(0), { 3, 3 });

        ov::op::v0::Interpolate::Attributes attributes;
        attributes.axes = ov::AxisSet{ 2, 3 };
        attributes.mode = "nearest";
        attributes.align_corners = false;
        attributes.antialias = false;
        attributes.pads_begin = { 0 };
        attributes.pads_end = { 0 };
        const auto outputShape = ov::opset1::Constant::create(
            ov::element::i64, ov::Shape{ 2 },
            ov::Shape{
                inputShape[2].is_dynamic() ? 9ul : static_cast<size_t>(inputShape[2].get_length()),
                inputShape[3].is_dynamic() ? 9ul : static_cast<size_t>(inputShape[3].get_length()) });
        intermediateOp = std::make_shared<ov::opset1::Interpolate>(pooling->output(0), outputShape, attributes);
        intermediateOp->set_friendly_name("intermediate");
    } else {
        intermediateOp = fakeQuantize1;
    }

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize2->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ov::opset1::AvgPool>(
        concat,
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("avgPool");

    ov::ResultVector results{
        std::make_shared<ov::opset1::Result>(avgPool),
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateWithConstantTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithReshapeAtTheEndTransformation(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const FakeQuantizeOnDataWithConstant& fqOnData3) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ov::opset1::Concat> concat1 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat1->set_friendly_name("concat1");

    const std::shared_ptr<Node> intermediate = makeMaxPool(concat1->output(0), {1ul, 1ul});

    const auto input3 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input3->set_friendly_name("input3");
    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const std::shared_ptr<ov::opset1::Concat> concat2 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{ fakeQuantize3, intermediate }, 1);
    concat2->set_friendly_name("concat2");

    const Shape concat2Shape = concat2->output(0).get_shape();
    const std::shared_ptr<Node> maxPool = makeMaxPool(concat2->output(0), {concat2Shape[2], concat2Shape[3]});
    const std::shared_ptr<Node> reshape = std::make_shared<ov::opset1::Reshape>(
        maxPool,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{2ul}, std::vector<size_t>{0, 0}),
        true);
    reshape->set_friendly_name("output");


    ov::ResultVector results{std::make_shared<ov::opset1::Result>(reshape)};

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2, input3 },
        "OriginalWithReshapeAtTheEndTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getOriginalWithIntermediateReshape(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::Shape& reshapeOutputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    const auto reshape1 = std::make_shared<ov::opset1::Reshape>(
        fakeQuantize1,
        ov::opset1::Constant::create(ov::element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
        true);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    const auto reshape2 = std::make_shared<ov::opset1::Reshape>(
        fakeQuantize2,
        ov::opset1::Constant::create(ov::element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
        true);
    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
            ov::OutputVector{ reshape1->output(0), reshape2->output(0) }, 1);
    concat->set_friendly_name("output");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(concat) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
            results,
            ov::ParameterVector{ input1, input2 },
            "ConcatWithIntermediateReshapeTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReference(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations& dequantizationOperations) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Concat>>(
        ov::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const std::shared_ptr<ov::Node> lastDequantization = makeDequantization(concat, dequantizationOperations);
    lastDequantization->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastDequantization) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    if (fqOnData1.outputPrecision != fqOnData2.outputPrecision) {
        throw std::runtime_error("FakeQuantize expected precisions are different");
    }
    const ov::element::Type fqOnDataPrecision = fqOnData1.outputPrecision;
    if (fqOnDataPrecision != ov::element::dynamic) {
        if (fakeQuantize1->get_output_element_type(0) != fakeQuantize2->get_output_element_type(0)) {
            throw std::runtime_error("FakeQuantize operation precisions are different");
        }
        const ov::element::Type fakeQuantizePrecision = fakeQuantize1->get_output_element_type(0);

        if (fqOnDataPrecision != fakeQuantizePrecision) {
            ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, fqOnDataPrecision);
            ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, fqOnDataPrecision);
            ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, fqOnDataPrecision);
        }
    }

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::get(
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const DequantizationOperations::Convert& convert1,
    const DequantizationOperations& dequantization1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const DequantizationOperations::Convert& convert2,
    const DequantizationOperations& dequantization2,
    const std::vector<ov::Any>& concatAttributes,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const std::int64_t& axis,
    const bool addNotPrecisionPreservedOperation) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input1->set_friendly_name("input1");

    std::shared_ptr<Node> parent1;
    if (fqOnData1.empty()) {
        parent1 = input1;
    } else {
        std::shared_ptr<Node> fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, inputPrecision, fqOnData1);
        fakeQuantize1->set_friendly_name("fakeQuantize1");
        parent1 = fakeQuantize1;
    }
    if (!convert1.empty()) {
        parent1 = std::make_shared<ov::opset1::Convert>(parent1, convert1.outPrecision);
    }
    if (!dequantization1.empty()) {
        parent1 = makeDequantization(parent1, dequantization1);
    }

    const auto input2 = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input2->set_friendly_name("input2");

    std::shared_ptr<Node> parent2;
    if (fqOnData2.empty()) {
        parent2 = input2;
    } else {
        std::shared_ptr<Node> fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, inputPrecision, fqOnData2);
        fakeQuantize2->set_friendly_name("fakeQuantize2");
        parent2 = fakeQuantize2;
    }
    if (!convert2.empty()) {
        parent2 = std::make_shared<ov::opset1::Convert>(parent2, convert2.outPrecision);
    }
    if (!dequantization2.empty()) {
        parent2 = makeDequantization(parent2, dequantization2);
    }

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{ parent1, parent2 }, axis);
    concat->set_friendly_name("concat");
    addAttributes({ concat }, concatAttributes);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const auto lastDequantization = makeDequantization(concat, dequantizationAfter);

    std::shared_ptr<ov::Node> parent = lastDequantization;
    if (addNotPrecisionPreservedOperation) {
        auto avgPool = std::make_shared<ov::opset1::AvgPool>(lastDequantization,
                                                             Strides{1, 1},
                                                             Shape{1, 1},
                                                             Shape{1, 1},
                                                             Shape{2, 2},
                                                             true,
                                                             ov::op::RoundingType::FLOOR);
        parent = avgPool;
    }

    parent->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::get(
    const ov::element::Type inputPrecision,
    const ov::Shape& inputShape1,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const DequantizationOperations::Convert& convert1,
    const DequantizationOperations& dequantization1,
    const bool addReshape1,
    const ov::Shape& inputShape2,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const DequantizationOperations::Convert& convert2,
    const DequantizationOperations& dequantization2,
    const bool addReshape2,
    const std::vector<ov::Any>& concatAttributes,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const std::int64_t& axis,
    const bool addNotPrecisionPreservedOperation) {
    const auto createReshape = [](const std::shared_ptr<Node>& parent) -> std::shared_ptr<Node> {
        const auto originalShape = parent->output(0).get_shape();
        std::vector<int> intermediateShape(originalShape.size());
        std::fill(intermediateShape.begin(), intermediateShape.end(), 1);
        intermediateShape[0] = ov::shape_size(originalShape);

        const auto reshape1 = std::make_shared<ov::opset1::Reshape>(
            parent,
            std::make_shared<ov::opset1::Constant>(ov::element::i32,
                                                   Shape{intermediateShape.size()},
                                                   intermediateShape),
            true);

        const auto maxPool = std::make_shared<ov::opset1::MaxPool>(reshape1,
                                                                   Strides{1, 1},
                                                                   Shape{1, 1},
                                                                   Shape{0, 0},
                                                                   Shape{2, 2},
                                                                   ov::op::RoundingType::FLOOR);

        const auto reshape2 = std::make_shared<ov::opset1::Reshape>(
            maxPool,
            std::make_shared<ov::opset1::Constant>(ov::element::i32, Shape{originalShape.size()}, originalShape),
            true);

        return reshape2;
    };

    const auto input1 = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape1);
    input1->set_friendly_name("input1");

    std::shared_ptr<Node> parent1;
    {
        if (fqOnData1.empty()) {
            parent1 = input1;
        } else {
            std::shared_ptr<Node> fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, inputPrecision, fqOnData1);
            fakeQuantize1->set_friendly_name("fakeQuantize1");
            parent1 = fakeQuantize1;
        }
        if (!convert1.empty()) {
            parent1 = std::make_shared<ov::opset1::Convert>(parent1, convert1.outPrecision);
        }
        if (!dequantization1.empty()) {
            parent1 = makeDequantization(parent1, dequantization1);
        }
        if (addReshape1) {
            parent1 = createReshape(parent1);
        }
    }

    const auto input2 = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape2);
    input2->set_friendly_name("input2");

    std::shared_ptr<Node> parent2;
    {
        if (fqOnData2.empty()) {
            parent2 = input2;
        } else {
            std::shared_ptr<Node> fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, inputPrecision, fqOnData2);
            fakeQuantize2->set_friendly_name("fakeQuantize2");
            parent2 = fakeQuantize2;
        }
        if (!convert2.empty()) {
            parent2 = std::make_shared<ov::opset1::Convert>(parent2, convert2.outPrecision);
        }
        if (!dequantization2.empty()) {
            parent2 = makeDequantization(parent2, dequantization2);
        }
        if (addReshape2) {
            parent2 = createReshape(parent2);
        }
    }

    std::shared_ptr<Node> parent;
    parent = std::make_shared<ov::opset1::Concat>(ov::OutputVector{ parent1, parent2 }, axis);
    parent->set_friendly_name("concat");
    addAttributes({ parent }, concatAttributes);

    auto& rtInfo = parent->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    parent = makeConvolution(parent, ov::element::f32, false);

    if (!dequantizationAfter.empty()) {
        parent = makeDequantization(parent, dequantizationAfter);
    }

    if (addNotPrecisionPreservedOperation) {
        auto avgPool = std::make_shared<ov::opset1::AvgPool>(parent,
                                                             Strides{1, 1},
                                                             Shape{1, 1},
                                                             Shape{1, 1},
                                                             Shape{2, 2},
                                                             true,
                                                             ov::op::RoundingType::FLOOR);
        parent = avgPool;
    }
    parent->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithNeighbors(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2,
    const std::string& neighborType,
    const std::string& additionalLayer) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const auto concat1 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { deqBefore1, deqBefore2 },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = "concat1";

    ov::ParameterVector inputs{input1, input2};
    std::shared_ptr<Node> mainBranch = concat1;
    std::string output_name1 = "concat1";
    auto deqCopy1 = dequantizationOperations1;
    if (additionalLayer == "convolution") {
        if (!deqCopy1.subtract.empty()) {
            DequantizationOperations deqSubtract;
            deqSubtract.subtract = deqCopy1.subtract;
            mainBranch = makeDequantization(mainBranch, deqSubtract);
            deqCopy1.subtract.erase();
        }
        ov::Shape convShape(4);
        convShape[1] = inputShape[1].get_length() + inputShape[1].get_length();
        convShape[0] = convShape[1] * 2;
        convShape[2] = convShape[3] = 1;
        auto convolutionAddition = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(mainBranch, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(ov::opset1::Constant::create(ov::element::i8, convShape, {1}),
                                               ov::element::f32)
                .get(),
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
        convolutionAddition->set_friendly_name("convolution_addition");
        output_name1 = "convolution_addition";
        mainBranch = convolutionAddition;
    }
    std::shared_ptr<Node> neighbor = fakeQuantize2;
    auto deqCopy2 = dequantizationOperations2;
    std::string output_name2 = "concat2";
    if (neighborType == "concat") {
        const auto input3 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
        input3->set_friendly_name("input3");
        inputs.push_back(input3);

        const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
        ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
        fakeQuantize3->set_friendly_name("fakeQuantize3");
        const auto deqBefore3 = makeDequantization(fakeQuantize3, dequantizationBefore);

        const auto concat2 = std::make_shared<ov::opset1::Concat>(
                ov::OutputVector { deqBefore2, deqBefore3 },
                1ull);
        concat2->set_friendly_name("concat2");
        auto& rtInfo2 = concat2->get_rt_info();
        rtInfo2["Variant::std::string"] = "concat2";

        neighbor = concat2;
    } else if (neighborType == "convolution") {
        if (!deqCopy2.subtract.empty()) {
            DequantizationOperations deqSubtract;
            deqSubtract.subtract = deqCopy2.subtract;
            neighbor = makeDequantization(neighbor, deqSubtract);
            deqCopy2.subtract.erase();
        }
        ov::Shape convShape(4);
        convShape[0] = inputShape[1].get_length() * 2;
        convShape[1] = inputShape[1].get_length();
        convShape[2] = convShape[3] = 1;
        auto convolutionNeighbor = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(neighbor, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(ov::opset1::Constant::create(ov::element::i8, convShape, {1}),
                                               ov::element::f32)
                .get(),
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
        convolutionNeighbor->set_friendly_name("convolution_neighbor");
        output_name2 = "convolution_neighbor";
        neighbor = convolutionNeighbor;
    }

    const std::shared_ptr<ov::Node> lastDequantization1 = makeDequantization(mainBranch, deqCopy1);
    lastDequantization1->set_friendly_name(output_name1);

    const std::shared_ptr<ov::Node> lastDequantization2 = makeDequantization(neighbor, deqCopy2);
    lastDequantization2->set_friendly_name(output_name2);

    const ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(lastDequantization1),
        std::make_shared<ov::opset1::Result>(lastDequantization2)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        inputs,
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithIntermediate(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    const size_t channels = inputShape[1].is_dynamic() ? 3ul : inputShape[1].get_length();
    ov::PartialShape inputShape1 = inputShape;

    if (inputShape1[2].is_static() && transparentIntermediate) {
        inputShape1[2] = inputShape1[2].get_length() - 2;
    }

    if (inputShape1[3].is_static() && transparentIntermediate) {
        inputShape1[3] = inputShape1[3].get_length() - 2;
    }

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore1);

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(deqBefore2, { 3, 3 });
    } else {
        const auto weights = ov::opset1::Constant::create(
            precision,
            ov::Shape{ channels, channels, 1, 1 },
            std::vector<float>(channels * channels, 1));

        intermediateOp = std::make_shared<ov::opset1::Convolution>(
            deqBefore2,
            weights,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { deqBefore1, intermediateOp },
        1);
    concat->set_friendly_name("concat");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const std::shared_ptr<ov::Node> lastDequantization1 = makeDequantization(concat, dequantizationAfter1);
    lastDequantization1->set_friendly_name("concat");

    const std::shared_ptr<ov::Node> lastDequantization2 = makeDequantization(intermediateOp, dequantizationAfter2);

    auto weights = ov::opset1::Constant::create(precision, ov::Shape{ channels, channels, 1, 1 }, { 1 });
    auto convolution = std::make_shared<ov::opset1::Convolution>(
        lastDequantization2,
        weights,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(lastDequantization1),
        std::make_shared<ov::opset1::Result>(convolution)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithIntermediateAvgPool(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    ov::PartialShape inputShape1 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    if (inputShape1[2].is_static()) {
        inputShape1[2] = inputShape[2].get_length() - 2;
    }
    if (inputShape1[3].is_static()) {
        inputShape1[3] = inputShape[3].get_length() - 2;
    }

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore2);

    std::shared_ptr<Node> intermediateOp  = makeMaxPool(deqBefore2, { 3, 3 });
    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { deqBefore1, intermediateOp },
        1);
    concat->set_friendly_name("concat");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const std::shared_ptr<ov::Node> parent1 = makeDequantization(concat, dequantizationAfter1);
    parent1->set_friendly_name("concat");

    std::shared_ptr<Node> parent2 = std::make_shared<ov::op::TypeRelaxed<ov::opset1::AvgPool>>(
        std::vector<ov::element::Type>{element::f32, element::f32},
        std::vector<ov::element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(intermediateOp, element::f32).get(),
        Strides{1, 1},
        Shape{1, 1},
        Shape{0, 0},
        Shape{2, 2},
        true,
        op::RoundingType::FLOOR);
    parent2->set_friendly_name("avgPool");

    parent2 = makeDequantization(parent2, dequantizationAfter2);

    ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(parent1),
        std::make_shared<ov::opset1::Result>(parent2)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithSplitedIntermediate(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ov::element::Type precisionAfterOperation,
    const bool addConvolution,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    size_t numSplit = 2;
    size_t splitedAxis = 1;

    auto inputShape1 = inputShape;
    inputShape1[1] = inputShape1[1].get_length() / numSplit;

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionAfterOperation);
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);


    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    replace_node(
        fakeQuantize2->get_input_node_shared_ptr(3),
        ov::pass::low_precision::NetworkHelper::toScalarIfPossible(fakeQuantize2->get_input_node_shared_ptr(3)));
    replace_node(
        fakeQuantize2->get_input_node_shared_ptr(4),
        ov::pass::low_precision::NetworkHelper::toScalarIfPossible(fakeQuantize2->get_input_node_shared_ptr(4)));

    fakeQuantize2->set_friendly_name("fakeQuantize2");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionAfterOperation);
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore1);

    std::shared_ptr<ov::op::Op> intermediateOp;

    const auto constant = std::make_shared<ov::opset1::Constant>(ov::element::i64, Shape{}, splitedAxis);
    intermediateOp = std::make_shared<ov::opset1::Split>(deqBefore2, constant, numSplit);
    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ deqBefore1, intermediateOp->output(0) }, splitedAxis);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const auto lastDequantization1 = makeDequantization(concat, dequantizationOperations1);
    const auto lastDequantization2 = makeDequantization(intermediateOp->output(1), dequantizationOperations2);
    lastDequantization1->set_friendly_name("output_1");

    ov::Output<Node> lastOutput = lastDequantization2;
    if (addConvolution) {
        auto weights = ov::opset1::Constant::create(
            precision,
            ov::Shape{ static_cast<size_t>(inputShape[1].get_length() / numSplit),
                           static_cast<size_t>(inputShape[1].get_length() / numSplit), 1, 1 }, { 1 });

        auto convolution = std::make_shared<ov::opset1::Convolution>(
            lastDequantization2,
            weights,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
        convolution->set_friendly_name("output_2");
        lastOutput = convolution->output(0);
    } else {
        lastOutput.get_node_shared_ptr()->set_friendly_name("output_2.1");
    }

    ov::ResultVector results{
        std::make_shared<ov::opset1::Result>(lastDequantization1),
        std::make_shared<ov::opset1::Result>(lastOutput)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceSelectionWithIntermediate(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore2);

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(deqBefore2, { 3, 3 });
    } else {
        auto weights = ov::opset1::Constant::create(
            precision,
            ov::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ov::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ov::Strides{ 1, 1 },
            ov::CoordinateDiff{ 0, 0 },
            ov::CoordinateDiff{ 0, 0 },
            ov::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector { deqBefore1, intermediateOp->output(0) },
        1);
    concat->set_friendly_name("concat");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const std::shared_ptr<ov::Node> lastDequantization1 = dequantizationOperations1.empty() ?
        concat :
        makeDequantization(concat, dequantizationOperations1);
    lastDequantization1->set_friendly_name("concat");

    const std::shared_ptr<ov::Node> lastDequantization2 = dequantizationOperations2.empty() ?
        nullptr :
        makeDequantization(intermediateOp, dequantizationOperations2);

    auto weights = ov::opset1::Constant::create(precision, ov::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ov::opset1::Convolution>(
        lastDequantization2 == nullptr ? intermediateOp : lastDequantization2,
        weights,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ov::ResultVector results {
        std::make_shared<ov::opset1::Result>(lastDequantization1),
        std::make_shared<ov::opset1::Result>(convolution)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithStridedSlice(
    const ov::element::Type inputPrecision,
    const ov::PartialShape inputShape,
    const FakeQuantizeOnData& fq1,
    const FakeQuantizeOnData& fq2,
    const DequantizationOperations& deqBefore,
    const ov::element::Type precisionBeforeConcat,
    const ov::element::Type precisionAfterConcat,
    const bool ssBeforeConcat,
    const bool ssAfterConcat,
    const DequantizationOperations& deqAfter1,
    const DequantizationOperations& deqAfter2) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input, inputPrecision, fq1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeConcat);
    fakeQuantize1->set_friendly_name("FakeQuantize_1");

    std::shared_ptr<ov::Node> parent1 = fakeQuantize1;

    const size_t rank = inputShape.rank().get_length();
    if (ssBeforeConcat) {
        const auto beginParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>(rank, 0));

        const auto endParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>{ 0, inputShape[1].get_length() - 2, 0, 0 });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        parent1 = std::make_shared<ov::opset1::StridedSlice>(parent1, beginParam, endParam, beginMask, endMask);
        parent1->set_friendly_name("StridedSlice_1");
    }

    const auto dequantizationBefore = makeDequantization(fakeQuantize1, deqBefore);
    const auto clamp = std::make_shared<ov::opset1::Clamp>(dequantizationBefore, 0.0, 6.0);
    clamp->set_friendly_name("Clamp");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(clamp, inputPrecision, fq2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeConcat);
    fakeQuantize2->set_friendly_name("FakeQuantize_2");

    const auto concat = std::make_shared<ov::opset1::Concat>(NodeVector{ parent1, fakeQuantize2 }, 1);
    concat->set_friendly_name("Concat");

    ov::ResultVector results;
    if (ssAfterConcat) {
        const auto concatShape = concat->get_output_partial_shape(0);
        const auto beginParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>(rank, 0));

        const auto endParam = ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{ rank },
            std::vector<int64_t>{ 0, concatShape[1].get_length() - 2, 0, 0 });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        const auto stridedSlice = std::make_shared<ov::opset1::StridedSlice>(concat, beginParam, endParam, beginMask, endMask);
        stridedSlice->set_friendly_name("StridedSlice_2");

        const auto dequantizationAfter1 = makeDequantization(stridedSlice, deqAfter1);
        const auto result1 = std::make_shared<ov::opset1::Result>(dequantizationAfter1);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    } else {
        const auto dequantizationAfter1 = makeDequantization(concat, deqAfter1);
        const auto result1 = std::make_shared<ov::opset1::Result>(dequantizationAfter1);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    }

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

    const auto maxPool = std::make_shared<ov::opset1::MaxPool>(
        concat,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    maxPool->set_friendly_name("MaxPool");

    const auto dequantizationAfter2 = makeDequantization(maxPool, deqAfter2);

    const std::shared_ptr<Node> convolution = makeConvolution(dequantizationAfter2, inputPrecision, false);

    const auto result2 = std::make_shared<ov::opset1::Result>(convolution);
    result2->set_friendly_name("Result_2");
    results.push_back(result2);

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithDifferentPrecisionOnChildren(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool multiChannel,
    const std::int64_t axis,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore2);

    const auto concat = std::make_shared<ov::opset1::Concat>(OutputVector{ deqBefore1, deqBefore2 }, axis);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);
    concat->set_friendly_name("concat");

    const auto lastDequantization1 = makeDequantization(concat->output(0), dequantizationAfter1);

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ov::opset1::AvgPool>(
        lastDequantization1,
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("AvgPool");

    ov::ResultVector results;
    results.push_back(std::make_shared<ov::opset1::Result>(avgPool));

    const std::shared_ptr<ov::opset1::MaxPool> maxPool = std::make_shared<ov::opset1::MaxPool>(
        concat->output(0),
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);

    const std::shared_ptr<ov::Node> lastDequantization2 = makeDequantization(maxPool, dequantizationAfter2);
    lastDequantization2->set_friendly_name("MaxPool");
    results.push_back(std::make_shared<ov::opset1::Result>(lastDequantization2));

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithIntermediateWithConstant(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ov::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const ov::element::Type precisionAfterDequantization) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, precisionBeforeOp);

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, precisionBeforeOp);

    std::shared_ptr<Node> intermediateOp;

    if (transparentIntermediate) {
        const auto deqBefore = makeDequantization(fakeQuantize1->output(0), dequantizationBefore);
        const auto pooling = makeMaxPool(fakeQuantize1->output(0), { 3, 3 });

        ov::op::v0::Interpolate::Attributes attributes;
        attributes.axes = ov::AxisSet{ 2, 3 };
        attributes.mode = "nearest";
        attributes.align_corners = false;
        attributes.antialias = false;
        attributes.pads_begin = { 0 };
        attributes.pads_end = { 0 };

        const auto outputShape = ov::opset1::Constant::create(
            ov::element::i64, ov::Shape{ 2 },
            ov::Shape{
                inputShape[2].is_dynamic() ? 9ul : static_cast<size_t>(inputShape[2].get_length()),
                inputShape[3].is_dynamic() ? 9ul : static_cast<size_t>(inputShape[3].get_length())});
        intermediateOp = std::make_shared<ov::opset1::Interpolate>(pooling->output(0), outputShape, attributes);
        intermediateOp->set_friendly_name("intermediate");
    } else {
        intermediateOp = fakeQuantize1;
    }

    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize2->output(0), intermediateOp->output(0) },
        1);
    concat->set_friendly_name("concat");
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const auto deqAfter = makeDequantization(concat->output(0), dequantizationAfter);
    deqAfter->set_friendly_name("concat");

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ov::opset1::AvgPool>(
        deqAfter,
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("avgPool");

    ov::ResultVector results{
        std::make_shared<ov::opset1::Result>(avgPool)
    };

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithReshapeAtTheEndTransformation(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const FakeQuantizeOnDataWithConstant& fqOnData3,
    const ov::element::Type precisionBeforeOp,
    const ov::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ov::opset1::Concat> concat1 = std::make_shared<ov::opset1::Concat>(
        ov::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat1, precisionAfterOperation);
    concat1->set_friendly_name("concat1");

    std::shared_ptr<Node> intermediate = makeMaxPool(concat1->output(0), {1ul, 1ul});

    const auto input3 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape));
    input3->set_friendly_name("input3");

    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const std::shared_ptr<ov::opset1::Concat> concat2 = std::make_shared<ov::opset1::Concat>(ov::OutputVector{ fakeQuantize3, intermediate }, 1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(concat2, precisionAfterOperation);
    concat2->set_friendly_name("concat2");

    const Shape concat2Shape = concat2->output(0).get_shape();
    const std::shared_ptr<Node> maxPool = makeMaxPool(concat2->output(0), {concat2Shape[2], concat2Shape[3]});
    const std::shared_ptr<Node> reshape = std::make_shared<ov::opset1::Reshape>(
        maxPool,
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{2ul}, std::vector<size_t>{0, 0}),
        true);
    reshape->set_friendly_name("output_original");

    const auto dequantization = makeDequantization(reshape->output(0), dequantizationOperations);
    dequantization->set_friendly_name("output");

    ov::ResultVector results{std::make_shared<ov::opset1::Result>(dequantization)};

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ input1, input2, input3 },
        "ReferenceWithReshapeAtTheEndTransformation");

    return function;
}

std::shared_ptr<ov::Model> ConcatFunction::getReferenceWithIntermediateReshape(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::Shape& reshapeOutputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const DequantizationOperations& dequantizationAfter) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, ov::element::u8);
    const auto reshape1 = std::make_shared<ov::opset1::Reshape>(
        fakeQuantize1,
        ov::opset1::Constant::create(ov::element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
        true);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, ov::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, ov::element::u8);
    const auto reshape2 = std::make_shared<ov::opset1::Reshape>(
        fakeQuantize2,
        ov::opset1::Constant::create(ov::element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
        true);
    const std::shared_ptr<ov::opset1::Concat> concat = std::make_shared<ov::opset1::Concat>(
            ov::OutputVector{ reshape1->output(0), reshape2->output(0) }, 1);
    concat->set_friendly_name("output_original");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = "concat";

    const auto dequantization = makeDequantization(concat, dequantizationAfter);
    dequantization->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(dequantization) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
            results,
            ov::ParameterVector{ input1, input2 },
            "ConcatWithIntermediateReshapeTransformation");

    return function;
}

std::shared_ptr<Node> ConcatFunction::makeMaxPool(const ov::Output<Node>& parent, const std::vector<size_t>& kernel) {
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;
    const auto pooling = std::make_shared<ov::opset1::MaxPool>(
        parent,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    return pooling;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
