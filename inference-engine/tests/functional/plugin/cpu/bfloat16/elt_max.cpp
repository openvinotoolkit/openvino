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

class Elt_max : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
//         Power (FP32)
//              |
//           Conv(BF16)  Const(FP32)
//              |        /
//        Eltwise(MAX)(FP32)
//              |
//            Conv(BF16)

        // STAGE1: construction of the GRAPH
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        auto channelsCount = inputShapes[1];
        const size_t conv0OutputChannels = 1;

        // add
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> powerConst = nullptr;
        if (netPrecision == Precision::FP32) {
            powerConst = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            powerConst = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto powerNode0 = std::make_shared<opset1::Multiply>(input1, powerConst);
        powerNode0->set_friendly_name("Power_0");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode0 = nullptr, weightsNode1 = nullptr;
        ngraph::Shape convFilterShape0 = { conv0OutputChannels, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        ngraph::Shape convFilterShape1 = { 1, conv0OutputChannels, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32_0, weightValuesFP32_1;
            weightValuesFP32_0.resize(conv0OutputChannels * channelsCount * 3 * 3);
            weightValuesFP32_1.resize(1 * conv0OutputChannels * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32_0.data(), weightValuesFP32_0.size());
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32_1.data(), weightValuesFP32_1.size());
            weightsNode0 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape0, weightValuesFP32_0);
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape1, weightValuesFP32_1);
        } else {
            std::vector<short> weightValuesBF16_0, weightValuesBF16_1;
            weightValuesBF16_0.resize(conv0OutputChannels * channelsCount * 3 * 3);
            weightValuesBF16_1.resize(1 * conv0OutputChannels * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16_0.data(), weightValuesBF16_0.size());
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16_1.data(), weightValuesBF16_1.size());
            weightsNode0 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape0, weightValuesBF16_0.data());
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape1, weightValuesBF16_1.data());
        }

        std::shared_ptr<ngraph::Node> convNode0 = std::make_shared<ngraph::opset1::Convolution>(
                powerNode0, weightsNode0,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode0->set_friendly_name("Convolution_0");

        // Eltwise, i.e. Max
        std::shared_ptr<ngraph::opset1::Constant> maxConst = nullptr;
        auto batchSize = inputShapes[0];
        auto heightSize = inputShapes[2];
        auto widthSize = inputShapes[3];
        if (netPrecision == Precision::FP32) {
            maxConst = opset1::Constant::create(ntype, Shape{batchSize, conv0OutputChannels, heightSize, widthSize}, { 2.0f });
        } else {
            maxConst = opset1::Constant::create(ntype, Shape{batchSize, conv0OutputChannels, heightSize, widthSize},
                    { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        maxConst->set_friendly_name("Max_const");
        auto eltMaxNode = std::make_shared<opset1::Maximum>(convNode0, maxConst);
        eltMaxNode->set_friendly_name("Elt_max");

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
                eltMaxNode, weightsNode1,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("Convolution_1");

        return std::make_shared<ngraph::Function>(convNode1, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE2: set up safe threshold <= 5% from maximum value of output tensor
        threshold = 0.2f;  // Max in fp32 network by output: 20.0761

        // STAGE3:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Convolution_0"] = "BF16";
        expectedPrecisions["Convolution_1"] = "BF16";
    }
};

TEST_P(Elt_max, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};


INSTANTIATE_TEST_SUITE_P(smoke_FP32_bfloat16_NoReshape, Elt_max,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({1, 3, 40, 40})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Elt_max::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BF16_bfloat16_NoReshape, Elt_max,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({1, 3, 40, 40})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Elt_max::getTestCaseName);
}  // namespace LayerTestsDefinitions
