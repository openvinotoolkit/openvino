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

class TopKInputsI32 : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision)override {
        //      Power   (FP32)
        //        |
        //      Convolution1 (BF16)       Const (I32)
        //               |                |
        //               \                /
        //                  TopK (FP32)
        //              (BF16)/        \ (I32)
        //                   |
        //         Convolution 2

        // STAGE1: construction of the GRAPH

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // multiply
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{1, 3, 40, 40});
        input1->set_friendly_name("Input_1");
        std::shared_ptr<ngraph::opset1::Constant> const1 = nullptr;
        if (netPrecision == Precision::FP32) {
            const1 = opset1::Constant::create(ntype, Shape{1}, { 2.0f });
        } else {
            const1 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(2.0f)) });
        }
        auto mulNode = std::make_shared<opset1::Multiply>(input1, const1);

        // add
        std::shared_ptr<ngraph::opset1::Constant> const2 = nullptr;
        if (netPrecision == Precision::FP32) {
            const2 = opset1::Constant::create(ntype, Shape{1}, { 1.0f });
        } else {
            const2 = opset1::Constant::create(ntype, Shape{1}, { bfloat16::from_bits(BFloat16Helpers::reducePrecisionBitwiseS(1.0f)) });
        }
        auto addNode = std::make_shared<opset1::Add>(mulNode, const2);
        addNode->set_friendly_name("Add_4");

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;
        ngraph::Shape convFilterShape = { 16, 3, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(16 * 3 * 3 * 3);
            BFloat16Helpers::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(16 * 3 * 3 * 3);
            BFloat16Helpers::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode = std::make_shared<ngraph::opset1::Convolution>(
            addNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode->set_friendly_name("Convolution_1");

        // TopK
        const auto k = make_shared<op::Constant>(element::i32, Shape{}, vector<int>{1});
        size_t axis = 1;
        ngraph::op::v1::TopK::Mode mode = ngraph::op::v1::TopK::Mode::MAX;
        ngraph::op::v1::TopK::SortType sort = ngraph::op::v1::TopK::SortType::NONE;
        auto argmaxNode = std::make_shared<opset1::TopK>(convNode, k, axis, mode, sort);
        argmaxNode->set_friendly_name("TopK_1");

        auto goe0 = make_shared<op::GetOutputElement>(argmaxNode, 0);
        auto goe1 = make_shared<op::GetOutputElement>(argmaxNode, 1);

        // convolution
        std::shared_ptr<ngraph::opset1::Constant> weightsNode2 = nullptr;
        ngraph::Shape convFilterShape2 = { 1, 1, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(1 * 1 * 3 * 3);
            BFloat16Helpers::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(1 * 1 * 3 * 3);
            BFloat16Helpers::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode2 = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape2, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
            goe0, weightsNode2,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("Convolution_2");

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{convNode2, goe1}, ngraph::ParameterVector{input1});
    }
    void SetUp()override  {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        threshold = 0.14f;  // max value in the latest tensor for FP32 network is 22.6

        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Add_4"] = "FP32";
        expectedPrecisions["Convolution_1"] = "BF16";
        expectedPrecisions["Convolution_2"] = "BF16";
        expectedPrecisions["TopK_1"] = "FP32";
    }
};

TEST_P(TopKInputsI32, CompareWithRefImpl) {
    test();
};


INSTANTIATE_TEST_CASE_P(FP32_bfloat16_NoReshape, TopKInputsI32,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        TopKInputsI32::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BF16_bfloat16_NoReshape, TopKInputsI32,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::BF16),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        TopKInputsI32::getTestCaseName);

}  // namespace LayerTestsDefinitions
