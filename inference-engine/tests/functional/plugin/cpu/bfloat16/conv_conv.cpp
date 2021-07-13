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

class ConvConv : public BasicBF16Test {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //     ScaleShift (FP32)
        //          |
        //        Conv (BF16)
        //          |
        //        Conv (BF16)

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // multiply
        auto input1 = std::make_shared<opset1::Parameter>(ngraph::element::f32, ngraph::Shape{inputShapes});
        auto const1 = opset1::Constant::create(ngraph::element::f32, Shape{1}, { 2.0f });
        auto mulNode = std::make_shared<opset1::Multiply>(input1, const1);

        // add
        auto const2 = opset1::Constant::create(ngraph::element::f32, Shape{1}, { 1.0f });
        auto addNode = std::make_shared<opset1::Add>(mulNode, const2);
        addNode->set_friendly_name("ADD_1");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;

        auto channelsCount = inputShapes[1];

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

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
            addNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
            ngraph::CoordinateDiff({ 1, 1 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("CONV_1");

        // Convolution
        ngraph::Shape convFilterShape2 = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        std::vector<float> weightValues2;
        weightValues2.resize(channelsCount * channelsCount * 3 * 3);
        FuncTestUtils::fillInputsBySinValues(weightValues2.data(), weightValues2.size());
        auto weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, convFilterShape2, weightValues2);
        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
            convNode1, weightsNode2,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("CONV_2");

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{convNode2}, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE1:
        threshold = 1.0f;  // Max in fp32 network by output CONV_2: 49.3427
        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["ADD_1"] = "ndef";
        expectedPrecisions["CONV_1"] = "BF16";
        expectedPrecisions["CONV_2"] = "BF16";
    }
};

TEST_P(ConvConv, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};

INSTANTIATE_TEST_SUITE_P(smoke_FP32_bfloat16_NoReshape, ConvConv,
                        ::testing::Combine(
                        ::testing::Values(Precision::FP32),
                        ::testing::Values(Precision::FP32),
                        ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                        ::testing::Values(SizeVector()),
                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvConv::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BF16_bfloat16_NoReshape, ConvConv,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvConv::getTestCaseName);


}  // namespace LayerTestsDefinitions
