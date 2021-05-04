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

#include "functional_test_utils/blob_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

class Faster100_5_1_1_Conv : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //                     Power (FP32)
        //                       |
        //                     Convolution (BF16)

        // STAGE1: constructin og the GRAPH
        auto channelsCount = inputShapes[1];

        // multiply
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
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
        addNode->set_friendly_name("Add_4");

        // problematic convolution: 100x5x1x1
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;
        ngraph::Shape convFilterShape = { channelsCount, channelsCount, 1, 1 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValues;
            weightValues.resize(channelsCount * channelsCount * 1 * 1, 0.f);
            weightValues[0] = 1.0f;
            weightValues[7] = 1.0f;
            weightValues[11] = 1.0f;
            weightValues[19] = 1.0f;
            weightValues[23] = 1.0f;
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, convFilterShape, weightValues);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 1 * 1, FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(0.0f));
            weightValuesBF16[0] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(1.0f);
            weightValuesBF16[7] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(1.0f);
            weightValuesBF16[11] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(1.0f);
            weightValuesBF16[19] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(1.0f);
            weightValuesBF16[23] = FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(1.0f);
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode = std::make_shared<ngraph::opset1::Convolution>(
            addNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode->set_friendly_name("Convolution_6");


        // ReLU
        auto reluNode = std::make_shared<opset1::Relu>(convNode);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{reluNode}, ngraph::ParameterVector{input1});
    }

    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Add_4"] = "ndef";
        expectedPrecisions["Convolution_6"] = "BF16";
    }
};

TEST_P(Faster100_5_1_1_Conv, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};


INSTANTIATE_TEST_CASE_P(smoke_bfloat16_NoReshape, Faster100_5_1_1_Conv,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(SizeVector({ 10, 5, 1, 1 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            Faster100_5_1_1_Conv::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, Faster100_5_1_1_Conv,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::BF16),
                            ::testing::Values(SizeVector({ 10, 5, 1, 1 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Faster100_5_1_1_Conv::getTestCaseName);


}  // namespace LayerTestsDefinitions
