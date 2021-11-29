// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bfloat16_helpers.hpp"

#include <memory>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {

class MobileNet_ssd_with_branching : public BasicBF16Test {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //                scaleshift
        //                    |
        //                   Conv1 (FP32)
        //                  |           \
        //               Conv2 (BF16)    \
        //                |              |
        //               relu(fused)     |
        //                |          Normalize (not LRN)
        //           Conv (DW)(BF16)     |
        //                |              |
        //               ReLU (Fused)    |
        //                  \           /
        //                    Concat

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        auto channelsCount = inputShapes[1];

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
        addNode->set_friendly_name("ADD_1");

        // Conv1
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

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
            addNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
            ngraph::CoordinateDiff({ 1, 1 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("CONV_1");

        // Conv2
        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
            convNode1, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
            ngraph::CoordinateDiff({ 1, 1 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("CONV_2");

        // ReLU
        auto reluNode =  std::make_shared<opset1::Relu>(convNode2);
        reluNode->set_friendly_name("RELU_2");

        // DW convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode2 = nullptr;
        ngraph::Shape convFilterShape2 = { channelsCount, 1, 1, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValues2FP32;
            weightValues2FP32.resize(channelsCount * 1 * 1 * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValues2FP32.data(), weightValues2FP32.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValues2FP32);
        } else {
            std::vector<short> weightValues2BF16;
            weightValues2BF16.resize(channelsCount * 1 * 1 * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValues2BF16.data(), weightValues2BF16.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValues2BF16.data());
        }

        std::shared_ptr<ngraph::Node> dwConvNode = std::make_shared<ngraph::opset1::GroupConvolution>(
            reluNode, weightsNode2,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
            ngraph::CoordinateDiff({ 1, 1 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        dwConvNode->set_friendly_name("DW_CONV");

        // ReLU
        auto reluNode2 =  std::make_shared<opset1::Relu>(dwConvNode);
        reluNode2->set_friendly_name("RELU_DW");

        // normalize
        const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2});
        float eps{1e-6f};
        auto eps_mode = op::EpsMode::ADD;

        auto normNode =  std::make_shared<opset1::NormalizeL2>(convNode1, axes, eps, eps_mode);
        normNode->set_friendly_name("NORM_1");

        // Concat
        ngraph::NodeVector concInputNodes = { reluNode2, normNode };
        auto concNode = std::make_shared<opset1::Concat>(concInputNodes, 1);
        concNode->set_friendly_name("CONC_1");

        return std::make_shared<ngraph::Function>(concNode, ngraph::ParameterVector{input1});
    }

    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE1:
        threshold = 0.85f;  // max value in latest tensor is 87.67
        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["ADD_1"] = "FP32";
        expectedPrecisions["CONV_1"] = "BF16";
        expectedPrecisions["RELU_2"] = "ndef";
        expectedPrecisions["DW_CONV"] = "BF16";
        expectedPrecisions["RELU_DW"] = "ndef";
    }
};

TEST_P(MobileNet_ssd_with_branching, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};

INSTANTIATE_TEST_CASE_P(smoke_FP32_bfloat16_NoReshape, MobileNet_ssd_with_branching,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MobileNet_ssd_with_branching::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, MobileNet_ssd_with_branching,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::BF16),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MobileNet_ssd_with_branching::getTestCaseName);

}  // namespace LayerTestsDefinitions
