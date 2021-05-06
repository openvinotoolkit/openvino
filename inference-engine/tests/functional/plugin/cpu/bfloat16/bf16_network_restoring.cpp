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

class BF16NetworkRestore1 : public BasicBF16Test  {
protected:
    std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) override {
        //   +   Power1(FP32)
        //        |
        //   +  AvgPooling1(BF16)
        //        |
        //   + Convolution1(BF16)
        //        |
        //   +    ReLU1(Fused)
        //        |------------------------
        //        |                        \
        //   +   Convolution2(BF16)      Convolution 3 (BF16)
        //           |                     /              \
        //   +        |                  ReLU2(FP32)     Normalize (FP32)
        //            \              /                      |
        //              Eltwise (Fused to Conv2)     ------/
        //                |                         /
        //              ReLU3  (Fused to Conv2)   /
        //                |                     /
        //             MaxPooling1 (BF16)      /
        //                   \            /
        //                      Eltwise
        //                         |


        // STAGE1: construction of the GRAPH

        ngraph::element::Type ntype = (netPrecision == Precision::FP32) ? ngraph::element::f32 : ngraph::element::bf16;
        // multiply
        auto input1 = std::make_shared<opset1::Parameter>(ntype, ngraph::Shape{1, 3, 224, 224});
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
        addNode->set_friendly_name("Power1");

        // AvgPooling
        auto avgpoolNode = std::make_shared<opset1::AvgPool>(addNode,
                                                             Strides{1, 1},
                                                             Shape{1, 1},
                                                             Shape{1, 1},
                                                             Shape{2, 2},
                                                             true,
                                                             op::RoundingType::FLOOR);
        avgpoolNode->set_friendly_name("AvgPooling1");

        // convolution1
        std::shared_ptr<ngraph::opset1::Constant> weightsNode = nullptr;
        ngraph::Shape convFilterShape = { 3, 3, 3, 3 };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(3 * 3 * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(3 * 3 * 3 * 3);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode = std::make_shared<ngraph::opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<ngraph::Node> convNode1 = std::make_shared<ngraph::opset1::Convolution>(
            avgpoolNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode1->set_friendly_name("Convolution1");

        // ReLU1
        auto reluNode = std::make_shared<opset1::Relu>(convNode1);
        reluNode->set_friendly_name("ReLU1");

        // convolution2
        std::shared_ptr<ngraph::Node> convNode2 = std::make_shared<ngraph::opset1::Convolution>(
            reluNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode2->set_friendly_name("Convolution2");

        // convolution3
        std::shared_ptr<ngraph::Node> convNode3 = std::make_shared<ngraph::opset1::Convolution>(
            reluNode, weightsNode,
            ngraph::Strides({ 1, 1 }),   // strides
            ngraph::CoordinateDiff({ 0, 0 }),  // pad begin
            ngraph::CoordinateDiff({ 0, 0 }),   // pad end
            ngraph::Strides({ 1, 1 }),        // dilation
            ngraph::op::PadType::EXPLICIT);   // pad type
        convNode3->set_friendly_name("Convolution3");

        // ReLU1
        auto reluNode2 = std::make_shared<opset1::Relu>(convNode3);
        reluNode2->set_friendly_name("ReLU2");

        // Norm1
        // normalize
        const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2});
        float eps{1e-6f};
        auto eps_mode = op::EpsMode::ADD;

        auto normNode =  std::make_shared<opset1::NormalizeL2>(convNode3, axes, eps, eps_mode);
        normNode->set_friendly_name("Norm1");



        // Eltwise1
        auto eltNode1 = std::make_shared<opset1::Add>(convNode2, reluNode2);
        eltNode1->set_friendly_name("Eltwise1");

        // ReLU3
        auto reluNode3 = std::make_shared<opset1::Relu>(eltNode1);
        reluNode3->set_friendly_name("ReLU3");

        // maxPooling1
        auto maxPoolNode = std::make_shared<opset1::MaxPool>(reluNode3,
                                                             Strides{1, 1},
                                                             Shape{1, 1},
                                                             Shape{0, 0},
                                                             Shape{2, 2},
                                                             op::RoundingType::FLOOR);
        maxPoolNode->set_friendly_name("maxPooling1");

        // Eltwise2
        auto eltNode2 = std::make_shared<opset1::Add>(maxPoolNode, normNode);
        eltNode2->set_friendly_name("Eltwise2");

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{eltNode2}, ngraph::ParameterVector{input1});
    }
    void SetUp() override {
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        threshold = 0.4f;  // max value in the latest tensor for FP32 network is 10.83

        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Power1"] = "FP32";
        expectedPrecisions["AvgPooling1"] = "BF16";
        expectedPrecisions["Convolution1"] = "BF16";
        expectedPrecisions["ReLU1"] = "ndef";
        expectedPrecisions["Convolution2"] = "BF16";
        expectedPrecisions["Convolution3"] = "BF16";
        expectedPrecisions["ReLU2"] = "BF16";
        expectedPrecisions["Norm1"] = "BF16";
        expectedPrecisions["Eltwise1"] = "ndef";
        expectedPrecisions["ReLU3"] = "ndef";
        expectedPrecisions["maxPooling1"] = "BF16";
        expectedPrecisions["Eltwise2"] = "BF16";
    }
};

TEST_P(BF16NetworkRestore1, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    test();
};


INSTANTIATE_TEST_CASE_P(smoke_BF16_bfloat16_NoReshape, BF16NetworkRestore1,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::BF16),
                            ::testing::Values(SizeVector({ 1, 3, 224, 224 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        BF16NetworkRestore1::getTestCaseName);

}  // namespace LayerTestsDefinitions
