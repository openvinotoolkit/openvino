// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <utility>

#include <ie_core.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

class ScaleshiftConv_x3_Eltwise : public BasicBF16Test {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //        scaleshift (FP32)
        //
        //        /        \
        //
        //       Conv1 (BF16)    Conv2 (BF16)
        //
        //        \        /
        //
        //        Eltwise (Fused to Conv1)
        //
        //          |
        //
        //         Conv3 (BF16)

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        auto channelsCount = inputShapes[1];
        const size_t outChannels = 16;

        // multiply
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> const1 = nullptr;
        if (netPrecision == Precision::FP32) {
            const1 = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            const1 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto mulNode = std::make_shared<opset1::Multiply>(input1, const1);

        // add
        std::shared_ptr<ngraph::opset1::Constant> const2 = nullptr;
        if (netPrecision == Precision::FP32) {
            const2 = opset1::Constant::create(ntype, Shape{1}, { 1.0f });
        } else {
            const2 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(1.0f)) });
        }
        auto addNode = std::make_shared<opset1::Add>(mulNode, const2);
        addNode->set_friendly_name("Add_1");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;
        ngraph::Shape convFilterShape = { outChannels, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(outChannels * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(outChannels * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
            addNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("Convolution_1");
        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
            addNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("Convolution_2");

        // Eltwise, i.e. Add
        auto eltNode = std::make_shared<opset1::Add>(convNode1, convNode2);
        eltNode->set_friendly_name("ELT_1");


        // Convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode3 = nullptr;
        ngraph::Shape convFilterShape3 = { outChannels, outChannels, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(outChannels * outChannels * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape3, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(outChannels * outChannels * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape3, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode3 = std::make_shared<ngraph::opset1::Convolution>(
            eltNode, weightsNode3,
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

        // STAGE1:
        threshold = 2.0f;  // max value in the latest tensor for FP32 network is 93.3

        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Add_1"] = "ndef";
        expectedPrecisions["Convolution_1"] = "BF16";
        expectedPrecisions["Convolution_2"] = "BF16";
        expectedPrecisions["ELT_1"] = "ndef";
        expectedPrecisions["Convolution_3"] = "BF16";
    }
};

TEST_P(ScaleshiftConv_x3_Eltwise, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};

INSTANTIATE_TEST_CASE_P(smoke_FP32_bfloat16_NoReshape, ScaleshiftConv_x3_Eltwise,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ScaleshiftConv_x3_Eltwise::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, ScaleshiftConv_x3_Eltwise,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::BF16),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ScaleshiftConv_x3_Eltwise::getTestCaseName);
}  // namespace LayerTestsDefinitions
