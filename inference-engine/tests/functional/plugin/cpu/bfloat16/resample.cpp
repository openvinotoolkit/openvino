// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>

#include <ie_core.hpp>
#include <ie_plugin_config.hpp>

#include "common_test_utils/common_utils.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

class Resample : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //    Convolution  (BF16)
        //        |
        //       Interpolation (Resample in the case of  mode = "nearest")   (FP32)
        //        |
        //     Convolution     (BF16)

        // STAGE1: construction of the GRAPH

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // add
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        auto channelsCount = inputShapes[1];
        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> addConst = nullptr;
        if (netPrecision == Precision::FP32) {
            addConst = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            addConst = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode = std::make_shared<opset1::Multiply>(input1, addConst);
        addNode->set_friendly_name("Add_1");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode1 = nullptr, weightsNode2 = nullptr;
        ngraph::Shape convFilterShape = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
                addNode, weightsNode1,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("Convolution_1");

        // interpolation
        auto heightSize = static_cast<long>(inputShapes[2]);
        auto weigthSize = static_cast<long>(inputShapes[3]);
        std::vector<int64_t> outShape = {2 * heightSize, 2 * weigthSize};

        auto interpolShape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, outShape);
        ngraph::op::InterpolateAttrs attrs;
        attrs.pads_begin.push_back(0);
        attrs.pads_end.push_back(0);
        attrs.axes = ngraph::AxisSet{2, 3};
        attrs.align_corners = false;
        attrs.mode = "nearest";
        attrs.antialias = false;
        auto interpolNode = std::make_shared<opset1::Interpolate>(
                convNode1,
                interpolShape, attrs);
        interpolNode->set_friendly_name("Interp");

        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
                interpolNode, weightsNode2,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("Convolution_2");
        return std::make_shared<ngraph::Function>(convNode2, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE2: set up safe threshold <= 5% from maximum value of output tensor
        threshold = 0.02f;  // Max in fp32 network by output: 2.35926

        // STAGE3:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Convolution_1"] = "BF16";
        expectedPrecisions["Interp"] = "FP32";
        expectedPrecisions["Convolution_2"] = "BF16";
    }
};

TEST_P(Resample, CompareWithRefImpl) {
    test();
};


INSTANTIATE_TEST_CASE_P(FP32_bfloat16_NoReshape, Resample,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({ 1, 1, 2, 2 })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Resample::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BF16_bfloat16_NoReshape, Resample,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({ 1, 1, 2, 2 })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Resample::getTestCaseName);

}  // namespace LayerTestsDefinitions
