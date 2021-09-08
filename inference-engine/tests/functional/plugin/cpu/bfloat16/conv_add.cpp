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

class ConvAdd : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
//              Power (FP32)
//                |
//            Conv(BF16)
//                |
//            Eltwise (SUM)(BF16)
//                |
//            Conv (BF16)

        auto channelsCount = inputShapes[1];

        // STAGE1: construction of the GRAPH
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // add
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> eltConst0 = nullptr, eltConst1 = nullptr;
        if (netPrecision == Precision::FP32) {
            eltConst0 = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
            eltConst1 = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            eltConst0 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
            eltConst1 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode0 = std::make_shared<opset1::Multiply>(input1, eltConst0);
        addNode0->set_friendly_name("Add_0");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode0 = nullptr, weightsNode1 = nullptr;
        ngraph::Shape convFilterShape = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode0 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode0 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode0 = std::make_shared<ngraph::opset1::Convolution>(
                addNode0, weightsNode0,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode0->set_friendly_name("Convolution_0");

        // eltwise, i.e. sum
        auto eltSumNode = std::make_shared<opset1::Add>(convNode0, eltConst1);
        eltSumNode->set_friendly_name("Elt_sum");

        // convolution
        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
                eltSumNode, weightsNode1,
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

        // 256 channels
        // threshold = 0.26f;  // Max in fp32 network by output: 5.26852

        // 3 channels
        threshold = 0.2f;  // Max in fp32 network by output: 4.90418

        // STAGE3:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Elt_sum"] = "ndef";
    }
};

TEST_P(ConvAdd, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};

INSTANTIATE_TEST_SUITE_P(smoke_FP32_bfloat16_NoReshape, ConvAdd,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({1, 3, 38, 38})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvAdd::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BF16_bfloat16_NoReshape, ConvAdd,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({1, 3, 38, 38})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvAdd::getTestCaseName);

}  // namespace LayerTestsDefinitions
