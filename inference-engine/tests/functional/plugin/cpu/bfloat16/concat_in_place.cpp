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
namespace {
    static const size_t inputSize = 2, concatAxe = 1;
    static std::vector<SizeVector> paramVector = {
            SizeVector({ 1, 1, inputSize, inputSize }),
            SizeVector({ 1, 2, inputSize, inputSize }),
            SizeVector({ 1, 3, inputSize, inputSize }),
            SizeVector({ 1, 4, inputSize, inputSize }),
            SizeVector({ 1, 5, inputSize, inputSize }),
            SizeVector({ 1, 6, inputSize, inputSize }),
            SizeVector({ 1, 7, inputSize, inputSize }),
            SizeVector({ 1, 8, inputSize, inputSize }),
            SizeVector({ 1, 9, inputSize, inputSize }),
            SizeVector({ 1, 10, inputSize, inputSize })};
}  // namespace

class Concat_in_place : public BasicBF16Test {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //              scaleshift
        //             /         \
        //           Conv      Conv
        //             \       /
        //              concat
        //                |
        //               relu

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

        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
                addNode, weightsNode,
                ngraph::Strides({ 1, 1 }),   // strides
                ngraph::CoordinateDiff({ 1, 1 }),  // pad begin
                ngraph::CoordinateDiff({ 1, 1 }),   // pad end
                ngraph::Strides({ 1, 1 }),        // dilation
                ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("CONV_2");

        // Concat
        ngraph::NodeVector concInputNodes = { convNode1, convNode2 };

        auto concNode = std::make_shared<opset1::Concat>(concInputNodes, concatAxe);
        concNode->set_friendly_name("CONC_1_TEST");

        // ReLU
        auto reluNode =  std::make_shared<opset1::Relu>(concNode);
        reluNode->set_friendly_name("RELU_1");

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{reluNode}, ngraph::ParameterVector{input1});
    }

    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        // STAGE1:
        threshold = 10e-1;
        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["ADD_1"] = "ndef";
        expectedPrecisions["CONV_1"] = "BF16";
        expectedPrecisions["CONV_2"] = "BF16";
    }
};

TEST_P(Concat_in_place, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};

INSTANTIATE_TEST_CASE_P(smoke_FP32_bfloat16_NoReshape, Concat_in_place,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::FP32),
                                ::testing::ValuesIn(paramVector),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Concat_in_place::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, Concat_in_place,
                        ::testing::Combine(
                                ::testing::Values(Precision::FP32),
                                ::testing::Values(Precision::BF16),
                                ::testing::ValuesIn(paramVector),
                                ::testing::Values(SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        Concat_in_place::getTestCaseName);
}  // namespace LayerTestsDefinitions
