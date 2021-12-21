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
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

namespace LayerTestsDefinitions {
typedef std::tuple< Precision, SizeVector, string, size_t, CoordinateDiff, string> convEltwiseDepthwiseTestParamsSet;

class ConvEltwiseDepthwise :
    public testing::WithParamInterface<convEltwiseDepthwiseTestParamsSet>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    std::shared_ptr<Function> fnPtr;
    SizeVector inputShapes;
    std::map<string, string> expectedPrecisions;
    float threshold = 7e-2;
    Precision netPrecision;
    size_t kernel;
    CoordinateDiff pads;
    std::string mkldnnPrimitive;

protected:
    std::shared_ptr<Function> createGraph(InferenceEngine::Precision netPrecision) {
        //            scaleshift (FP32)
        //                |
        //               Conv (BF16)
        //                |
        //              Relu (Eltwise Fused into Conv)
        //                |
        //            scaleshift (Depthwise Fused into Conv)

        element::Type ntype = (netPrecision == Precision::FP32) ? element::f32 : element::bf16;
        size_t chCnt = inputShapes[1];

        // multiply
        auto input1 = std::make_shared<opset1::Parameter>(ntype, Shape{ inputShapes });
        input1->set_friendly_name("Input_1");
        std::shared_ptr<opset1::Constant> const1 = nullptr;
        if (netPrecision == Precision::FP32) {
            const1 = opset1::Constant::create(ntype, Shape{ 1 }, { 2.0f });
        } else {
            const1 = opset1::Constant::create(ntype, Shape{ 1 }, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto mulNode = std::make_shared<opset1::Multiply>(input1, const1);

        // add
        std::shared_ptr<opset1::Constant> const2 = nullptr;
        if (netPrecision == Precision::FP32) {
            const2 = opset1::Constant::create(ntype, Shape{ 1 }, { 1.0f });
        } else {
            const2 = opset1::Constant::create(ntype, Shape{ 1 }, { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(1.0f)) });
        }
        auto addNode = std::make_shared<opset1::Add>(mulNode, const2);
        addNode->set_friendly_name("SS_1");

        // convolution
        std::shared_ptr<opset1::Constant> weightsNode = nullptr;
        Shape convFilterShape = { chCnt, chCnt, kernel, kernel };  // out channel, /input channels, kernel h, kernel w
        if (netPrecision == Precision::FP32) {
            std::vector<float> weightValuesFP32;
            weightValuesFP32.resize(chCnt * chCnt * kernel * kernel);
            FuncTestUtils::fillInputsBySinValues(weightValuesFP32.data(), weightValuesFP32.size());
            weightsNode = std::make_shared<opset1::Constant>(ntype, convFilterShape, weightValuesFP32);
        } else {
            std::vector<short> weightValuesBF16;
            weightValuesBF16.resize(chCnt * chCnt * kernel * kernel);
            FuncTestUtils::fillInputsBySinValues(weightValuesBF16.data(), weightValuesBF16.size());
            weightsNode = std::make_shared<opset1::Constant>(ntype, convFilterShape, weightValuesBF16.data());
        }

        std::shared_ptr<Node> convNode1 = std::make_shared<opset1::Convolution>(
            addNode, weightsNode, Strides({ 1, 1 }), pads, pads, Strides({ 1, 1 }), op::PadType::EXPLICIT);
        convNode1->set_friendly_name("CONV");

        // Eltwise, i.e. Relu
        auto reluNode = std::make_shared<opset1::Relu>(convNode1);
        reluNode->set_friendly_name("RELU");

        // multiply
        std::shared_ptr<opset1::Constant> const3 = nullptr;
        if (netPrecision == Precision::FP32) {
            const3 = opset1::Constant::create(ntype, Shape{ 1, chCnt, 1, 1 }, { 3.0f });
        } else {
            const3 = opset1::Constant::create(ntype, Shape{ 1, chCnt, 1, 1 },
                    { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(3.0f)) });
        }
        auto mulNode2 = std::make_shared<opset1::Multiply>(reluNode, const3);

        // add
        std::shared_ptr<opset1::Constant> const4 = nullptr;
        if (netPrecision == Precision::FP32) {
            const4 = opset1::Constant::create(ntype, Shape{ 1, chCnt, 1, 1 }, { 2.0f });
        } else {
            const4 = opset1::Constant::create(ntype, Shape{ 1, chCnt, 1, 1 },
                    { bfloat16::from_bits(FuncTestUtils::Bf16TestUtils::reducePrecisionBitwiseS(2.0f)) });
        }
        auto addNode2 = std::make_shared<opset1::Add>(mulNode2, const4);
        addNode2->set_friendly_name("SS_2");

        return std::make_shared<Function>(NodeVector{ addNode2 }, ParameterVector{ input1 });
    }
public:
    static string getTestCaseName(testing::TestParamInfo<convEltwiseDepthwiseTestParamsSet> obj) {
        Precision netPrecision;
        SizeVector inputShapes;
        string targetDevice;
        size_t kernel;
        CoordinateDiff pads;
        string mkldnnPrimitive;
        std::tie(netPrecision, inputShapes, targetDevice, kernel, pads, mkldnnPrimitive) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "mkldnnPrimitive=" << mkldnnPrimitive << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void Run_test() {
        if (!InferenceEngine::with_cpu_x86_bfloat16()) {
            // on platforms which do not support bfloat16, we are disabling bf16 tests since there are no bf16 primitives,
            // tests are useless on such platforms
            return;
        }
        std::tie(netPrecision, inputShapes, targetDevice, kernel, pads, mkldnnPrimitive) = this->GetParam();
        InferenceEngine::CNNNetwork cnnNet(fnPtr);

        for (const auto& inputItem : cnnNet.getInputsInfo()) {
            inputItem.second->setPrecision(Precision::FP32);
        }

        string inputName = cnnNet.getInputsInfo().begin()->first;
        string outputName = cnnNet.getOutputsInfo().begin()->first;
        auto ie = InferenceEngine::Core();
        // BF16 inference
        std::map<string, string> options;
        if (netPrecision == InferenceEngine::Precision::FP32) {
            options[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] = InferenceEngine::PluginConfigParams::YES;
        } else {
            options[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] = InferenceEngine::PluginConfigParams::NO;
        }
        options[InferenceEngine::PluginConfigParams::KEY_PERF_COUNT] = InferenceEngine::PluginConfigParams::YES;

        auto exec_net1 = ie.LoadNetwork(cnnNet, targetDevice, options);
        auto req1 = exec_net1.CreateInferRequest();

        InferenceEngine::Blob::Ptr inBlob1 = req1.GetBlob(inputName);
        FuncTestUtils::fillInputsBySinValues(inBlob1);

        req1.Infer();
        auto outBlobBF16 = req1.GetBlob(outputName);
        InferenceEngine::MemoryBlob::CPtr mout1 = InferenceEngine::as<InferenceEngine::MemoryBlob>(outBlobBF16);
        ASSERT_NE(mout1, nullptr);
        auto lm1 = mout1->rmap();

        // FP32 infrence
        // if netPrecision is not eq to the FP32 - change network precision and recreate network
        InferenceEngine::CNNNetwork cnnNetFP32(createGraph(InferenceEngine::Precision::FP32));
        string inputNameFP32 = cnnNetFP32.getInputsInfo().begin()->first;
        string outputNameFP32 = cnnNetFP32.getOutputsInfo().begin()->first;
        for (const auto& inputItem : cnnNetFP32.getInputsInfo()) {
            inputItem.second->setPrecision(Precision::FP32);
        }
        auto exec_net2 = ie.LoadNetwork(cnnNetFP32, targetDevice,
            { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO } });
        auto req2 = exec_net2.CreateInferRequest();

        req2.SetBlob(inputNameFP32, inBlob1);

        req2.Infer();
        auto outBlobFP32 = req2.GetBlob(outputNameFP32);
        InferenceEngine::MemoryBlob::CPtr mout2 = InferenceEngine::as<InferenceEngine::MemoryBlob>(outBlobFP32);
        ASSERT_NE(mout2, nullptr);
        auto lm2 = mout2->rmap();

        FuncTestUtils::compareRawBuffers(lm1.as<const float*>(), lm2.as<const float*>(), mout1->size(), mout2->size(),
                                                         FuncTestUtils::CompareType::ABS_AND_REL,
                                                         threshold, threshold);

        // Stage2: verification of performance counters
        std::pair<string, string> wrongLayer =
            BFloat16Helpers::matchPerfCountPrecisionVsExpected(req1.GetPerformanceCounts(), expectedPrecisions);
        if (wrongLayer.first != string("")) {
            string layerInPerfCounts = wrongLayer.first + " " + wrongLayer.second;
            string layerExpected = wrongLayer.first + " " + expectedPrecisions[wrongLayer.first];
            ASSERT_EQ(layerInPerfCounts, layerExpected);
        }
        fnPtr.reset();
    }

    void SetUp() override {
        std::vector<size_t> inputShape;
        std::tie(netPrecision, inputShapes, targetDevice, kernel, pads, mkldnnPrimitive) = this->GetParam();
        fnPtr = createGraph(netPrecision);

        expectedPrecisions["SS_1"] = "FP32";
        expectedPrecisions["CONV"] = mkldnnPrimitive;
        expectedPrecisions["RELU"] = "ndef";
        expectedPrecisions["SS_2"] = "ndef";
    }
};

TEST_P(ConvEltwiseDepthwise, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run_test();
};

INSTANTIATE_TEST_SUITE_P(smoke_FP32_bfloat16_1x1_depthwise_BF16, ConvEltwiseDepthwise,
    ::testing::Combine(
        ::testing::Values(Precision::FP32),
        ::testing::Values(SizeVector({ 1, 5, 1, 1 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(size_t(1)),
        ::testing::Values(CoordinateDiff({ 0, 0 })),
        ::testing::Values(std::string("jit_avx512_1x1_BF16"))),
    ConvEltwiseDepthwise::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FP32_bfloat16_gemm_depthwise_BF16, ConvEltwiseDepthwise,
    ::testing::Combine(
        ::testing::Values(Precision::FP32),
        ::testing::Values(SizeVector({ 1, 3, 10, 10 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(size_t(3)),
        ::testing::Values(CoordinateDiff({ 1, 1 })),
        ::testing::Values(std::string("jit_avx512_BF16"))),
    ConvEltwiseDepthwise::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FP32_bfloat16_conv_depthwise_BF16, ConvEltwiseDepthwise,
    ::testing::Combine(
        ::testing::Values(Precision::FP32),
        ::testing::Values(SizeVector({ 1, 5, 10, 10 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(size_t(3)),
        ::testing::Values(CoordinateDiff({ 0, 0 })),
        ::testing::Values(std::string("jit_avx512_BF16"))),
    ConvEltwiseDepthwise::getTestCaseName);

}  // namespace LayerTestsDefinitions
