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

class Elt_x3 : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
//                       Power (FP32)
//                    /    |          \
//            Conv(BF16)   Conv(BF16)  Conv(BF16)
//                /        |          /
// ----------------------------------------------
//    Eltwise(MAX)(FP32)  Eltwise(Mul) (FP32)
//             |            |
//            Conv(BF16)   Conv(BF16)
//             \           /
//            Eltwise (SUM)(BF16)
//                |
//            Conv (BF16)

        auto channelsCount = inputShapes[1];

        // STAGE1: construction of the GRAPH
        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // add
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{inputShapes});
        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> addConst = nullptr;
        if (netPrecision == Precision::FP32) {
            addConst = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            addConst = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode0 = std::make_shared<opset1::Multiply>(input1, addConst);
        addNode0->set_friendly_name("Add_0");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode0_1 = nullptr, weightsNode0_2 = nullptr,
        weightsNode0_3 = nullptr, weightsNode1 = nullptr,
        weightsNode2 = nullptr, weightsNode3 = nullptr;
        ngraph::Shape convFilterShape = { channelsCount, channelsCount, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode0_1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
            weightsNode0_2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
            weightsNode0_3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
            weightsNode3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(channelsCount * channelsCount * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode0_1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
            weightsNode0_2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
            weightsNode0_3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
            weightsNode1 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
            weightsNode3 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode0_1 = std::make_shared<ngraph::opset1::Convolution>(
                addNode0, weightsNode0_1,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode0_1->set_friendly_name("Convolution_0_1");

        std::shared_ptr<ngraph::Node> convNode0_2 = std::make_shared<ngraph::opset1::Convolution>(
                addNode0, weightsNode0_2,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode0_2->set_friendly_name("Convolution_0_2");

        std::shared_ptr<ngraph::Node> convNode0_3 = std::make_shared<ngraph::opset1::Convolution>(
                addNode0, weightsNode0_3,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode0_3->set_friendly_name("Convolution_0_3");

        // Eltwise, i.e. Mul
        auto eltMulNode = std::make_shared<opset1::Multiply>(convNode0_1, convNode0_2);
        eltMulNode->set_friendly_name("Elt_mul");

        // Eltwise, i.e. Max
        std::shared_ptr<ngraph::opset1::Constant> maxConst = nullptr;
        if (netPrecision == Precision::FP32) {
            maxConst = opset1::Constant::create(ntype, Shape{inputShapes}, { 2.0f });
        } else {
            maxConst = opset1::Constant::create(ntype, Shape{inputShapes},
                    { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto eltMaxNode = std::make_shared<opset1::Maximum>(convNode0_3, maxConst);
        eltMaxNode->set_friendly_name("Elt_max");

        // convolution
        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
                eltMulNode, weightsNode1,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("Convolution_1");

        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
                eltMaxNode, weightsNode2,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("Convolution_2");

        // eltwise, i.e. sum
        auto eltSumNode = std::make_shared<opset1::Add>(convNode1, convNode2);
        eltSumNode->set_friendly_name("Elt_sum");

        // convolution
        std::shared_ptr<ngraph::Node> convNode3 = std::make_shared<ngraph::opset1::Convolution>(
                eltSumNode, weightsNode3,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode3->set_friendly_name("Convolution_3");

        return std::make_shared<ngraph::Function>(convNode3, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE2: set up safe threshold <= 5% from maximum value of output tensor

        // 256 channels, 38 x 38 size
        // threshold = 0.6f;  // Max in fp32 network by output: 12.0983

        // 3 channels, 4 x 4 size
        threshold = 30.6f;  // Max in fp32 network by output: 879.077

        // STAGE3:
        // filling of expected precision of layer execution defined by precision of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Convolution_1"] = "BF16";
        expectedPrecisions["Convolution_2"] = "BF16";
        expectedPrecisions["Convolution_3"] = "BF16";
        expectedPrecisions["Elt_sum"] = "ndef";
    }
};

TEST_P(Elt_x3, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};

INSTANTIATE_TEST_CASE_P(smoke_FP32_bfloat16_NoReshape, Elt_x3,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({1, 3, 4, 4})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Elt_x3::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, Elt_x3,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::Values(SizeVector({1, 3, 4, 4})),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Elt_x3::getTestCaseName);

}  // namespace LayerTestsDefinitions
