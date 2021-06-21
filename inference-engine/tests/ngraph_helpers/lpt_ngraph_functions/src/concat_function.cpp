// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/concat_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat->set_friendly_name("output");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat->set_friendly_name("output");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithChildAndOutput(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    std::shared_ptr<ngraph::opset1::Result> res1;
    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const auto concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat->set_friendly_name("110");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto clamp = std::make_shared<ngraph::opset1::Clamp>(concat, 0.0, 6.0);
    clamp->set_friendly_name("111");

    ResultVector results{ std::make_shared<ngraph::opset1::Result>(clamp), std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithChildAndOutputTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const std::string& neighborType,
    const std::string& additionalLayer) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { fakeQuantize1->output(0), fakeQuantize2->output(0) },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat1");

    ngraph::ParameterVector inputs{input1, input2};

    ngraph::ResultVector results { };
    if (additionalLayer == "convolution") {
        auto convShape = inputShape;
        convShape[1] += convShape[1];
        convShape[0] = convShape[1] * 2;
        convShape[2] = convShape[3] = 1;
        auto convolutionAddition = std::make_shared<ngraph::opset1::Convolution>(
                concat1,
                std::make_shared<opset1::Multiply>(
                        std::make_shared<opset1::Convert>(opset1::Constant::create(element::i8, convShape, {1}), element::f32),
                        opset1::Constant::create(element::f32, Shape{}, {1})),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
        convolutionAddition->set_friendly_name("convolution_addition");
        results.push_back(std::make_shared<ngraph::opset1::Result>(convolutionAddition));
    }
    if (neighborType == "concat") {
        const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
        input3->set_friendly_name("input3");
        const auto fakeQuantize3 = makeFakeQuantize(input3, precision, fqOnData3);
        fakeQuantize3->set_friendly_name("fakeQuantize3");
        inputs.push_back(input3);

        const auto concat2 = std::make_shared<ngraph::opset1::Concat>(
                ngraph::OutputVector { fakeQuantize2->output(0), fakeQuantize3->output(0) },
                1ull);
        concat2->set_friendly_name("concat2");
        auto& rtInfo2 = concat2->get_rt_info();
        rtInfo2["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat2");
        results.push_back(std::make_shared<ngraph::opset1::Result>(concat1));
        results.push_back(std::make_shared<ngraph::opset1::Result>(concat2));
    } else if (neighborType == "convolution") {
        auto convShape = inputShape;
        convShape[0] = convShape[1] * 2;
        convShape[2] = convShape[3] = 1;
        auto convolutionNeighbor = std::make_shared<ngraph::opset1::Convolution>(
                fakeQuantize2,
                std::make_shared<opset1::Multiply>(
                        std::make_shared<opset1::Convert>(opset1::Constant::create(element::i8, convShape, {1}), element::f32),
                        opset1::Constant::create(element::f32, Shape{}, {1})),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
        convolutionNeighbor->set_friendly_name("convolution_neighbor");
        results.push_back(std::make_shared<ngraph::opset1::Result>(convolutionNeighbor));
    }


    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        inputs,
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    } else {
        auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        intermediateOp,
        weights,
        ngraph::Strides { 1, 1 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::Strides { 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithIntermediateAvgPool(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const std::vector<size_t> inputShape1 = { inputShape[0], inputShape[1], inputShape[2] - 2, inputShape[3] - 2 };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    std::shared_ptr<Node> parent2 = std::make_shared<ngraph::opset1::AvgPool>(
        intermediateOp,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);
    parent2->set_friendly_name("avgPool");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(parent2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithSplitedIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const bool addConvolution) {
    size_t numSplit = 2;
    size_t splitedAxis = 1;


    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1] / numSplit,
        inputShape[2],
        inputShape[3]
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<ngraph::op::Op> intermediateOp;

    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    intermediateOp = std::make_shared<ngraph::opset1::Split>(fakeQuantize2->output(0), constant, numSplit);

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, splitedAxis);
    concat->set_friendly_name("output_1");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    Output<Node> lastOutput = intermediateOp->output(1);
    if (addConvolution) {
        auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1] / numSplit, inputShape[1] / numSplit, 1, 1 }, { 1 });
        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            intermediateOp->output(1),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
        lastOutput = convolution->output(0);
    }
    lastOutput.get_node_shared_ptr()->set_friendly_name("output_2");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(lastOutput),
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalSelectionWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(fakeQuantize2->output(0), { 3, 3 });
    } else {
        auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        intermediateOp,
        weights,
        ngraph::Strides { 1, 1 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::CoordinateDiff { 0, 0 },
        ngraph::Strides { 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(concat),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
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

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithStridedSlice(
    const ngraph::element::Type precision,
    const ngraph::Shape inputShape,
    const FakeQuantizeOnData& fq1,
    const FakeQuantizeOnData& fq2,
    const bool ssBeforeConcat,
    const bool ssAfterConcat) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantize(input, precision, fq1);
    fakeQuantize1->set_friendly_name("FakeQuantize_1");

    std::shared_ptr<ngraph::Node> parent1 = fakeQuantize1;

    if (ssBeforeConcat) {
        const auto beginParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ inputShape.size() },
            std::vector<int64_t>(inputShape.size(), 0));

        const auto endParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ inputShape.size() },
            std::vector<size_t>{ inputShape[0], inputShape[1] - 2ul, inputShape[2], inputShape[3] });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        parent1 = std::make_shared<ngraph::opset1::StridedSlice>(parent1, beginParam, endParam, beginMask, endMask);
        parent1->set_friendly_name("StridedSlice_1");
    }

    const auto clamp = std::make_shared<ngraph::opset1::Clamp>(fakeQuantize1, 0.0, 6.0);
    clamp->set_friendly_name("Clamp");
    const auto fakeQuantize2 = makeFakeQuantize(clamp, precision, fq2);
    fakeQuantize2->set_friendly_name("FakeQuantize_2");

    const auto concat = std::make_shared<ngraph::opset1::Concat>(NodeVector{ parent1, fakeQuantize2 }, 1);
    concat->set_friendly_name("Concat");


    ngraph::ResultVector results;
    if (ssAfterConcat) {
        const auto concatShape = concat->get_output_shape(0);
        const auto beginParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ concatShape.size() },
            std::vector<int64_t>(concatShape.size(), 0));

        const auto endParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ concatShape.size() },
            std::vector<size_t>{ concatShape[0], concatShape[1] - 2ul, concatShape[2], concatShape[3] });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        const auto stridedSlice = std::make_shared<ngraph::opset1::StridedSlice>(concat, beginParam, endParam, beginMask, endMask);
        stridedSlice->set_friendly_name("StridedSlice_2");

        const auto result1 = std::make_shared<ngraph::opset1::Result>(stridedSlice);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    } else {
        const auto result1 = std::make_shared<ngraph::opset1::Result>(concat);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    }

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

    const auto maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        concat,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    maxPool->set_friendly_name("MaxPool");

    const auto result2 = std::make_shared<ngraph::opset1::Result>(maxPool);
    result2->set_friendly_name("Result_2");
    results.push_back(result2);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithDifferentPrecisionOnChildren(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        concat->output(0),
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("AvgPool");

    const auto maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        concat->output(0),
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    maxPool->set_friendly_name("MaxPool");

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(avgPool));
    results.push_back(std::make_shared<ngraph::opset1::Result>(maxPool));

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithIntermediateWithConstant(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");


    std::shared_ptr<ngraph::op::Op> intermediateOp;

    if (transparentIntermediate) {
        const auto pooling = makeMaxPool(fakeQuantize1->output(0), { 3, 3 });

        ngraph::op::v0::InterpolateAttrs attributes;
        attributes.axes = ngraph::AxisSet{ 2, 3 };
        attributes.mode = "nearest";
        attributes.align_corners = false;
        attributes.antialias = false;
        attributes.pads_begin = { 0 };
        attributes.pads_end = { 0 };
        const auto outputShape = op::Constant::create(
            ngraph::element::i64, ngraph::Shape{ 2 },
            ngraph::Shape{ inputShape[2], inputShape[3] });
        intermediateOp = std::make_shared<ngraph::opset1::Interpolate>(pooling->output(0), outputShape, attributes);
        intermediateOp->set_friendly_name("intermediate");
    } else {
        intermediateOp = fakeQuantize1;
    }

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize2->output(0), intermediateOp->output(0) }, 1);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(concat),
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateWithConstantTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithReshapeAtTheEndTransformation(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const FakeQuantizeOnDataWithConstant& fqOnData3) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ngraph::opset1::Concat> concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    concat1->set_friendly_name("concat1");

    const std::shared_ptr<Node> intermediate = makeMaxPool(concat1->output(0), {1ul, 1ul});

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");
    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const std::shared_ptr<ngraph::opset1::Concat> concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ fakeQuantize3, intermediate }, 1);
    concat2->set_friendly_name("concat2");

    const Shape concat2Shape = concat2->output(0).get_shape();
    const std::shared_ptr<Node> maxPool = makeMaxPool(concat2->output(0), {concat2Shape[2], concat2Shape[3]});
    const std::shared_ptr<Node> reshape = std::make_shared<ngraph::opset1::Reshape>(
        maxPool,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2ul}, std::vector<size_t>{0, 0}),
        true);
    reshape->set_friendly_name("output");


    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape)};

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2, input3 },
        "OriginalWithReshapeAtTheEndTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getOriginalWithIntermediateReshape(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& reshapeOutputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantize(input1, precision, fqOnData1);
    const auto reshape1 = std::make_shared<opset1::Reshape>(
            fakeQuantize1,
            opset1::Constant::create(element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
            true);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantize(input2, precision, fqOnData2);
    const auto reshape2 = std::make_shared<opset1::Reshape>(
            fakeQuantize2,
            opset1::Constant::create(element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
            true);
    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
            ngraph::OutputVector{ reshape1->output(0), reshape2->output(0) }, 1);
    concat->set_friendly_name("output");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(concat) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{ input1, input2 },
            "ConcatWithIntermediateReshapeTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations& dequantizationOperations) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = ngraph::builder::subgraph::makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Concat>>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization = makeDequantization(concat, dequantizationOperations);
    lastDequantization->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    if (fqOnData1.outputPrecision != fqOnData2.outputPrecision) {
        throw std::runtime_error("FakeQuantize expected precisions are different");
    }
    const ngraph::element::Type fqOnDataPrecision = fqOnData1.outputPrecision;
    if (fqOnDataPrecision != ngraph::element::undefined) {
        if (fakeQuantize1->get_output_element_type(0) != fakeQuantize2->get_output_element_type(0)) {
            throw std::runtime_error("FakeQuantize operation precisions are different");
        }
        const ngraph::element::Type fakeQuantizePrecision = fakeQuantize1->get_output_element_type(0);

        if (fqOnDataPrecision != fakeQuantizePrecision) {
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, fqOnDataPrecision);
            ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, fqOnDataPrecision);
        }
    }

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::get(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const DequantizationOperations::Convert& convert1,
    const DequantizationOperations& dequantization1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const DequantizationOperations::Convert& convert2,
    const DequantizationOperations& dequantization2,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const std::int64_t& axis) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input1->set_friendly_name("input1");

    std::shared_ptr<Node> parent1 = makeFakeQuantizeTypeRelaxed(input1, inputPrecision, fqOnData1);
    if (!convert1.empty()) {
        parent1 = std::make_shared<opset1::Convert>(parent1, convert1.outPrecision);
    }
    if (!dequantization1.empty()) {
        parent1 = makeDequantization(parent1, dequantization1);
    }

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input2->set_friendly_name("input2");

    std::shared_ptr<Node> parent2 = makeFakeQuantizeTypeRelaxed(input2, inputPrecision, fqOnData2);
    if (!convert2.empty()) {
        parent2 = std::make_shared<opset1::Convert>(parent2, convert2.outPrecision);
    }
    if (!dequantization2.empty()) {
        parent2 = makeDequantization(parent2, dequantization2);
    }

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ parent1, parent2 }, axis);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto lastDequantization = makeDequantization(concat, dequantizationAfter);
    lastDequantization->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithNeighbors(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const FakeQuantizeOnData& fqOnData3,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2,
    const std::string& neighborType,
    const std::string& additionalLayer) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const auto concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, deqBefore2 },
        1ull);
    concat1->set_friendly_name("concat1");

    auto& rtInfo1 = concat1->get_rt_info();
    rtInfo1["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat1");

    ngraph::ParameterVector inputs{input1, input2};
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
        auto convShape = inputShape;
        convShape[1] += convShape[1];
        convShape[0] = convShape[1] * 2;
        convShape[2] = convShape[3] = 1;
        auto convolutionAddition = std::make_shared<op::TypeRelaxed<opset1::Convolution>>(
                element::TypeVector{ element::f32, element::f32 },
                element::TypeVector{ element::f32 },
                op::TemporaryReplaceOutputType(mainBranch, element::f32).get(),
                op::TemporaryReplaceOutputType(opset1::Constant::create(element::i8, convShape, {1}), element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
        convolutionAddition->set_friendly_name("convolution_addition");
        output_name1 = "convolution_addition";
        mainBranch = convolutionAddition;
    }
    std::shared_ptr<Node> neighbor = fakeQuantize2;
    auto deqCopy2 = dequantizationOperations2;
    std::string output_name2 = "concat2";
    if (neighborType == "concat") {
        const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        input3->set_friendly_name("input3");
        inputs.push_back(input3);

        const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
        low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
        fakeQuantize3->set_friendly_name("fakeQuantize3");
        const auto deqBefore3 = makeDequantization(fakeQuantize3, dequantizationBefore);

        const auto concat2 = std::make_shared<ngraph::opset1::Concat>(
                ngraph::OutputVector { deqBefore2, deqBefore3 },
                1ull);
        concat2->set_friendly_name("concat2");
        auto& rtInfo2 = concat2->get_rt_info();
        rtInfo2["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat2");

        neighbor = concat2;
    } else if (neighborType == "convolution") {
        if (!deqCopy2.subtract.empty()) {
            DequantizationOperations deqSubtract;
            deqSubtract.subtract = deqCopy2.subtract;
            neighbor = makeDequantization(neighbor, deqSubtract);
            deqCopy2.subtract.erase();
        }
        auto convShape = inputShape;
        convShape[0] = convShape[1] * 2;
        convShape[2] = convShape[3] = 1;
        auto convolutionNeighbor = std::make_shared<op::TypeRelaxed<ngraph::opset1::Convolution>>(
                element::TypeVector{ element::f32, element::f32 },
                element::TypeVector{ element::f32 },
                op::TemporaryReplaceOutputType(neighbor, element::f32).get(),
                op::TemporaryReplaceOutputType(opset1::Constant::create(element::i8, convShape, {1}), element::f32).get(),
                ngraph::Strides{ 1, 1 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::CoordinateDiff{ 0, 0 },
                ngraph::Strides{ 1, 1 });
        convolutionNeighbor->set_friendly_name("convolution_neighbor");
        output_name2 = "convolution_neighbor";
        neighbor = convolutionNeighbor;
    }

    const std::shared_ptr<ngraph::Node> lastDequantization1 = makeDequantization(mainBranch, deqCopy1);
    lastDequantization1->set_friendly_name(output_name1);

    const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(neighbor, deqCopy2);
    lastDequantization2->set_friendly_name(output_name2);

    const ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(lastDequantization2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        inputs,
        "ConcatWithNeighborsTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore1);

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(deqBefore2, { 3, 3 });
    } else {
        const auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            deqBefore2,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, intermediateOp },
        1);
    concat->set_friendly_name("concat");
    low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization1 = makeDequantization(concat, dequantizationAfter1);
    lastDequantization1->set_friendly_name("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(intermediateOp, dequantizationAfter2);

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        lastDequantization2,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithIntermediateAvgPool(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    const std::vector<size_t> inputShape1 = { inputShape[0], inputShape[1], inputShape[2] - 2, inputShape[3] - 2};
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore2);

    std::shared_ptr<Node> intermediateOp  = makeMaxPool(deqBefore2, { 3, 3 });
    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, intermediateOp },
        1);
    concat->set_friendly_name("concat");
    low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::shared_ptr<ngraph::Node> parent1 = makeDequantization(concat, dequantizationAfter1);
    parent1->set_friendly_name("concat");

    std::shared_ptr<Node> parent2 = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::AvgPool>>(
        std::vector<ngraph::element::Type>{ element::f32, element::f32 },
        std::vector<ngraph::element::Type>{ element::f32 },
        ngraph::op::TemporaryReplaceOutputType(intermediateOp, element::f32).get(),
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);
    parent2->set_friendly_name("avgPool");

    parent2 = makeDequantization(parent2, dequantizationAfter2);

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(parent1),
        std::make_shared<ngraph::opset1::Result>(parent2)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithSplitedIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ngraph::element::Type precisionAfterOperation,
    const bool addConvolution,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    size_t numSplit = 2;
    size_t splitedAxis = 1;

    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1] / numSplit,
        inputShape[2],
        inputShape[3]
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionAfterOperation);
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);


    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    replace_node(
        fakeQuantize2->get_input_node_shared_ptr(3),
        ngraph::pass::low_precision::NetworkHelper::toScalarIfPossible(fakeQuantize2->get_input_node_shared_ptr(3)));
    replace_node(
        fakeQuantize2->get_input_node_shared_ptr(4),
        ngraph::pass::low_precision::NetworkHelper::toScalarIfPossible(fakeQuantize2->get_input_node_shared_ptr(4)));

    fakeQuantize2->set_friendly_name("fakeQuantize2");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionAfterOperation);
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore1);

    std::shared_ptr<ngraph::op::Op> intermediateOp;

    const auto constant = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ }, splitedAxis);
    intermediateOp = std::make_shared<ngraph::opset1::Split>(deqBefore2, constant, numSplit);
    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ deqBefore1, intermediateOp->output(0) }, splitedAxis);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto lastDequantization1 = makeDequantization(concat, dequantizationOperations1);
    const auto lastDequantization2 = makeDequantization(intermediateOp->output(1), dequantizationOperations2);
    lastDequantization1->set_friendly_name("output_1");

    Output<Node> lastOutput = lastDequantization2;
    if (addConvolution) {
        auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1] / numSplit, inputShape[1] / numSplit, 1, 1 }, { 1 });

        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            lastDequantization2,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
        convolution->set_friendly_name("output_2");
        lastOutput = convolution->output(0);
    } else {
        lastOutput.get_node_shared_ptr()->set_friendly_name("output_2.1");
    }

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(lastOutput)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceSelectionWithIntermediate(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore1,
    const DequantizationOperations& dequantizationBefore2,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations1,
    const DequantizationOperations& dequantizationOperations2) {
    const std::vector<size_t> inputShape1 = {
        inputShape[0],
        inputShape[1],
        inputShape[2] - (transparentIntermediate ? 2 : 0),
        inputShape[3] - (transparentIntermediate ? 2 : 0)
    };

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore1);

    const std::vector<size_t> inputShape2 = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore2);

    std::shared_ptr<Node> intermediateOp;
    if (transparentIntermediate) {
        intermediateOp = makeMaxPool(deqBefore2, { 3, 3 });
    } else {
        auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
            std::vector<float>(inputShape[1] * inputShape[1], 1));

        intermediateOp = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantize2->output(0),
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    }

    intermediateOp->set_friendly_name("intermediate");

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector { deqBefore1, intermediateOp->output(0) },
        1);
    concat->set_friendly_name("concat");
    low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization1 = dequantizationOperations1.empty() ?
        concat :
        makeDequantization(concat, dequantizationOperations1);
    lastDequantization1->set_friendly_name("concat");

    const std::shared_ptr<ngraph::Node> lastDequantization2 = dequantizationOperations2.empty() ?
        nullptr :
        makeDequantization(intermediateOp, dequantizationOperations2);

    auto weights = ngraph::opset1::Constant::create(precision, ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 }, { 1 });
    auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        lastDequantization2 == nullptr ? intermediateOp : lastDequantization2,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution");

    ngraph::ResultVector results {
        std::make_shared<ngraph::opset1::Result>(lastDequantization1),
        std::make_shared<ngraph::opset1::Result>(convolution)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithStridedSlice(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape inputShape,
    const FakeQuantizeOnData& fq1,
    const FakeQuantizeOnData& fq2,
    const DequantizationOperations& deqBefore,
    const ngraph::element::Type precisionBeforeConcat,
    const ngraph::element::Type precisionAfterConcat,
    const bool ssBeforeConcat,
    const bool ssAfterConcat,
    const DequantizationOperations& deqAfter1,
    const DequantizationOperations& deqAfter2) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input, inputPrecision, fq1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeConcat);
    fakeQuantize1->set_friendly_name("FakeQuantize_1");

    std::shared_ptr<ngraph::Node> parent1 = fakeQuantize1;

    if (ssBeforeConcat) {
        const auto beginParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ inputShape.size() },
            std::vector<int64_t>(inputShape.size(), 0));

        const auto endParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ inputShape.size() },
            std::vector<size_t>{ inputShape[0], inputShape[1] - 2ul, inputShape[2], inputShape[3] });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        parent1 = std::make_shared<ngraph::opset1::StridedSlice>(parent1, beginParam, endParam, beginMask, endMask);
        parent1->set_friendly_name("StridedSlice_1");
    }

    const auto dequantizationBefore = makeDequantization(fakeQuantize1, deqBefore);
    const auto clamp = std::make_shared<ngraph::opset1::Clamp>(dequantizationBefore, 0.0, 6.0);
    clamp->set_friendly_name("Clamp");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(clamp, inputPrecision, fq2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeConcat);
    fakeQuantize2->set_friendly_name("FakeQuantize_2");

    const auto concat = std::make_shared<ngraph::opset1::Concat>(NodeVector{ parent1, fakeQuantize2 }, 1);
    concat->set_friendly_name("Concat");

    ngraph::ResultVector results;
    if (ssAfterConcat) {
        const auto concatShape = concat->get_output_shape(0);
        const auto beginParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ concatShape.size() },
            std::vector<int64_t>(concatShape.size(), 0));

        const auto endParam = ngraph::op::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{ concatShape.size() },
            std::vector<size_t>{ concatShape[0], concatShape[1] - 2ul, concatShape[2], concatShape[3] });

        const std::vector<int64_t> beginMask{ 1, 0, 1, 1 };
        const std::vector<int64_t> endMask{ 1, 0, 1, 1 };

        const auto stridedSlice = std::make_shared<ngraph::opset1::StridedSlice>(concat, beginParam, endParam, beginMask, endMask);
        stridedSlice->set_friendly_name("StridedSlice_2");

        const auto dequantizationAfter1 = makeDequantization(stridedSlice, deqAfter1);
        const auto result1 = std::make_shared<ngraph::opset1::Result>(dequantizationAfter1);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    } else {
        const auto dequantizationAfter1 = makeDequantization(concat, deqAfter1);
        const auto result1 = std::make_shared<ngraph::opset1::Result>(dequantizationAfter1);
        result1->set_friendly_name("Result_1");
        results.push_back(result1);
    }

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

    const auto maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        concat,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    maxPool->set_friendly_name("MaxPool");

    const auto dequantizationAfter2 = makeDequantization(maxPool, deqAfter2);

    const auto result2 = std::make_shared<ngraph::opset1::Result>(dequantizationAfter2);
    result2->set_friendly_name("Result_2");
    results.push_back(result2);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithDifferentPrecisionOnChildren(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool multiChannel,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter1,
    const DequantizationOperations& dequantizationAfter2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    const auto deqBefore1 = makeDequantization(fakeQuantize1, dequantizationBefore);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    const auto deqBefore2 = makeDequantization(fakeQuantize2, dequantizationBefore);

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ deqBefore1, deqBefore2 }, 1);
    low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);
    concat->set_friendly_name("concat");

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto lastDequantization1 = makeDequantization(concat->output(0), dequantizationAfter1);

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

    const auto avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        lastDequantization1,
        stride,
        padBegin,
        padEnd,
        kernel,
        true,
        roundingType,
        padType);
    avgPool->set_friendly_name("AvgPool");

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(avgPool));

    if (!dequantizationAfter2.empty()) {
        const std::shared_ptr<ngraph::opset1::MaxPool> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
            concat->output(0),
            stride,
            padBegin,
            padEnd,
            kernel,
            roundingType,
            padType);

        const std::shared_ptr<ngraph::Node> lastDequantization2 = makeDequantization(maxPool, dequantizationAfter2);
        lastDequantization2->set_friendly_name("MaxPool");
        results.push_back(std::make_shared<ngraph::opset1::Result>(lastDequantization2));
    }

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithDifferentChildrenTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithIntermediateWithConstant(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool transparentIntermediate,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const ngraph::element::Type precisionBeforeOp,
    const DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const ngraph::element::Type precisionAfterDequantization) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    fakeQuantize1->set_friendly_name("fakeQuantize1");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, precisionBeforeOp);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    fakeQuantize2->set_friendly_name("fakeQuantize2");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, precisionBeforeOp);

    std::shared_ptr<Node> intermediateOp;

    if (transparentIntermediate) {
        const auto deqBefore = makeDequantization(fakeQuantize1->output(0), dequantizationBefore);
        const auto pooling = makeMaxPool(fakeQuantize1->output(0), { 3, 3 });

        ngraph::op::v0::InterpolateAttrs attributes;
        attributes.axes = ngraph::AxisSet{ 2, 3 };
        attributes.mode = "nearest";
        attributes.align_corners = false;
        attributes.antialias = false;
        attributes.pads_begin = { 0 };
        attributes.pads_end = { 0 };

        const auto outputShape = op::Constant::create(
            ngraph::element::i64, ngraph::Shape{ 2 },
            ngraph::Shape{ inputShape[2], inputShape[3] });
        intermediateOp = std::make_shared<ngraph::opset1::Interpolate>(pooling->output(0), outputShape, attributes);
        intermediateOp->set_friendly_name("intermediate");
    } else {
        intermediateOp = fakeQuantize1;
    }

    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize2->output(0), intermediateOp->output(0) },
        1);
    concat->set_friendly_name("concat");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(concat, precisionAfterOperation);

    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto deqAfter = makeDequantization(concat->output(0), dequantizationAfter);
    deqAfter->set_friendly_name("concat");

    ngraph::ResultVector results{
        std::make_shared<ngraph::opset1::Result>(deqAfter)
    };

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "ConcatWithIntermediateTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithReshapeAtTheEndTransformation(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const FakeQuantizeOnDataWithConstant& fqOnData3,
    const ngraph::element::Type precisionBeforeOp,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationOperations) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input1->set_friendly_name("input1");

    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize1, precisionBeforeOp);
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input2->set_friendly_name("input2");

    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize2, precisionBeforeOp);
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const std::shared_ptr<ngraph::opset1::Concat> concat1 = std::make_shared<ngraph::opset1::Concat>(
        ngraph::OutputVector{ fakeQuantize1->output(0), fakeQuantize2->output(0) }, 1);
    low_precision::NetworkHelper::setOutDataPrecision(concat1, precisionAfterOperation);
    concat1->set_friendly_name("concat1");

    std::shared_ptr<Node> intermediate = makeMaxPool(concat1->output(0), {1ul, 1ul});

    const auto input3 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input3->set_friendly_name("input3");

    const auto fakeQuantize3 = makeFakeQuantizeTypeRelaxed(input3, precision, fqOnData3);
    low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(fakeQuantize3, precisionBeforeOp);
    fakeQuantize3->set_friendly_name("fakeQuantize3");

    const std::shared_ptr<ngraph::opset1::Concat> concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ fakeQuantize3, intermediate }, 1);
    low_precision::NetworkHelper::setOutDataPrecision(concat2, precisionAfterOperation);
    concat2->set_friendly_name("concat2");

    const Shape concat2Shape = concat2->output(0).get_shape();
    const std::shared_ptr<Node> maxPool = makeMaxPool(concat2->output(0), {concat2Shape[2], concat2Shape[3]});
    const std::shared_ptr<Node> reshape = std::make_shared<ngraph::opset1::Reshape>(
        maxPool,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2ul}, std::vector<size_t>{0, 0}),
        true);
    reshape->set_friendly_name("output_original");

    const auto dequantization = makeDequantization(reshape->output(0), dequantizationOperations);
    dequantization->set_friendly_name("output");

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(dequantization)};

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2, input3 },
        "ReferenceWithReshapeAtTheEndTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> ConcatFunction::getReferenceWithIntermediateReshape(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::Shape& reshapeOutputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const DequantizationOperations& dequantizationAfter) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    input1->set_friendly_name("input1");
    const auto fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input1, precision, fqOnData1);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize1, element::u8);
    const auto reshape1 = std::make_shared<opset1::Reshape>(
            fakeQuantize1,
            opset1::Constant::create(element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
            true);

    const std::vector<size_t> inputShape2 = inputShape;
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    input2->set_friendly_name("input2");
    const auto fakeQuantize2 = makeFakeQuantizeTypeRelaxed(input2, precision, fqOnData2);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize2, element::u8);
    const auto reshape2 = std::make_shared<opset1::Reshape>(
            fakeQuantize2,
            opset1::Constant::create(element::i64, Shape{reshapeOutputShape.size()}, reshapeOutputShape),
            true);
    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(
            ngraph::OutputVector{ reshape1->output(0), reshape2->output(0) }, 1);
    concat->set_friendly_name("output_original");
    auto& rtInfo = concat->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

    const auto dequantization = makeDequantization(concat, dequantizationAfter);
    dequantization->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantization) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{ input1, input2 },
            "ConcatWithIntermediateReshapeTransformation");

    return function;
}

std::shared_ptr<Node> ConcatFunction::makeMaxPool(const Output<Node>& parent, const std::vector<size_t>& kernel) {
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;
    const auto pooling = std::make_shared<ngraph::opset1::MaxPool>(
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
}  // namespace ngraph
