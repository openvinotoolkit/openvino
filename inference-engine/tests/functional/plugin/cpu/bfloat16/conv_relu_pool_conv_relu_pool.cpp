// Copyright (C) 2018-2021 Intel Corporation
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

class ConvReLUPoolConvReLUPool : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //    Convolution1  (FP32)
        //        |
        //       ReLU1      (Fused)
        //        |
        //     Pooling1     (BF16)
        //        |
        //    Convolution2  (BF16)
        //        |
        //       ReLU2      (Fused)
        //        |
        //     Pooling2     (BF16)
        //        |
        //    Convolution3  (BF16)


        // STAGE1: construction of the GRAPH

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        auto channelsCount = inputShapes[1];

        // multiply
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");

        // convolution1
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;
        ngraph::Shape convFilterShape = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode = std::make_shared<ngraph::opset1::Convolution>(
            input1, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode->set_friendly_name("Convolution_1");

        // ReLU
        auto reluNode = std::make_shared<opset1::Relu>(convNode);
        reluNode->set_friendly_name("ReLU_1");

        // Pooling
        auto avgpoolNode = std::make_shared<opset1::AvgPool>(reluNode,
                                                             Strides{1, 1},
                                                             Shape{1, 1},
                                                             Shape{1, 1},
                                                             Shape{2, 2},
                                                             true,
                                                             op::RoundingType::FLOOR);
        avgpoolNode->set_friendly_name("AvgPool_1");

        // convolution2
        std::shared_ptr<ngraph::opset1::Constant> weightsNode2 = nullptr;
        ngraph::Shape convFilterShape2 = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
            avgpoolNode, weightsNode2,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("Convolution_2");

        // ReLU
        auto reluNode2 = std::make_shared<opset1::Relu>(convNode2);
        reluNode2->set_friendly_name("ReLU_2");

        // Pooling
        auto maxpoolNode2 = std::make_shared<opset1::MaxPool>(reluNode2,
                                                             Strides{1, 1},
                                                             Shape{1, 1},
                                                             Shape{0, 0},
                                                             Shape{2, 2},
                                                             op::RoundingType::FLOOR);
        maxpoolNode2->set_friendly_name("MaxPool_2");

        // convolution3
        std::shared_ptr<ngraph::opset1::Constant> weightsNode3 = nullptr;
        ngraph::Shape convFilterShape3 = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape3, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape3, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode3 = std::make_shared<ngraph::opset1::Convolution>(
            maxpoolNode2, weightsNode3,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode3->set_friendly_name("Convolution_3");

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{convNode3}, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        threshold = 0.2f;  // max value in the latest tensor for FP32 network is 9.8

        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Convolution_1"] = "BF16";
        expectedPrecisions["ReLU_1"] = "ndef";
        expectedPrecisions["AvgPool_1"] = "BF16";
        expectedPrecisions["Convolution_2"] = "BF16";
        expectedPrecisions["ReLU_2"] = "ndef";
        expectedPrecisions["MaxPool_2"] = "BF16";
        expectedPrecisions["Convolution_3"] = "BF16";
    }
};

TEST_P(ConvReLUPoolConvReLUPool, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};


INSTANTIATE_TEST_CASE_P(smoke_FP32_bfloat16_NoReshape, ConvReLUPoolConvReLUPool,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvReLUPoolConvReLUPool::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, ConvReLUPoolConvReLUPool,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::BF16),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvReLUPoolConvReLUPool::getTestCaseName);

}  // namespace LayerTestsDefinitions
