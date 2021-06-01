// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include "ie_common.h"
#include <ie_blob.h>
#include <math.h>
#include <map>
#include <string>
#include <utility>
#include <memory>
#include <tuple>
#include <vector>

#include "ngraph/opsets/opset1.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include <ie_system_conf.h>

namespace LayerTestsDefinitions {

/**
 * class providing static helpers for bfloat16 functional tests
 * using functions you can fill the tensor content by some periodic law or compare
 *
 */
class BFloat16Helpers {
public:
    static std::pair<std::string, std::string> matchPerfCountPrecisionVsExpected(
        const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfCounts,
        const std::map<std::string, std::string>& expected) {
        for (auto e : expected) {
            auto it = perfCounts.find(e.first);
            if (it == perfCounts.end()) {
                return std::pair<std::string, std::string>(e.first, "NOT_FOUND_IN_PERF_COUNTS");
            }
            // get the latest n symbols by number of e.second
            std::string execType = it->second.exec_type;
            std::string pfPrecision = execType.substr(execType.length() - e.second.length(), e.second.length());
            if (pfPrecision != e.second) {
                return std::pair<std::string, std::string>(e.first, pfPrecision);
            }
        }
        return std::pair<std::string, std::string>("", "");
    }

    static float getMaxAbsValue(const float* data, size_t size) {
        float maxVal = 0.f;
        for (size_t i = 0; i < size; i++) {
            if (fabs(data[i]) > maxVal) {
                maxVal = fabs(data[i]);
            }
        }
        return maxVal;
    }

    static float reducePrecisionBitwise(const float in) {
        float f = in;
        int* i = reinterpret_cast<int*>(&f);
        int t2 = *i & 0xFFFF0000;
        float ft1 = *(reinterpret_cast<float*>(&t2));
        if ((*i & 0x8000) && (*i & 0x007F0000) != 0x007F0000) {
            t2 += 0x10000;
            ft1 = *(reinterpret_cast<float*>(&t2));
        }
        return ft1;
    }

    static short reducePrecisionBitwiseS(const float in) {
        float f = reducePrecisionBitwise(in);
        int intf = *reinterpret_cast<int*>(&f);
        intf = intf >> 16;
        short s = intf;
        return s;
    }
};


typedef std::tuple<
                   InferenceEngine::Precision,
                   InferenceEngine::Precision,
                   InferenceEngine::SizeVector,
                   InferenceEngine::SizeVector,
                   std::string> basicParams;


/**
 * Base class for bf16 tests
 * the flow in this test assume to load network in FP32 and in BF16 modes and verify
 * 1. difference between outptut's tensor with some treshold.
 * 2. which preciosion was selected for layers described in runtime info of performance counters
 *
 * To develop new test you need to
 * 1. define class inherriten from  BasicBF16Test and implement SetUp(). For example:
 *
 * class ScaleshiftConv_x3_Eltwise : public BasicBF16Test {
 * protected:
 * void SetUp() override {
 *  fnPtr = std::make_shared<ngraph::Function>(ngraph::NodeVector{convNode3}, ngraph::ParameterVector{input1});

        // STAGE1:
        threshold = 9e-1;

        // STAGE2:
        // filling of expected precision of layer execution defined by precisoin of input tensor to the primitive and reflected in
        // performance counters
        expectedPrecisions["Add_4"] = "FP32";
        expectedPrecisions["Convolution_6"] = "BF16";
        expectedPrecisions["Convolution_7"] = "BF16";
        expectedPrecisions["Add_8"] = "ndef";
 *      expectedPrecisions["Convolution_10"] = "BF16";
 *      }
 *      };
 *
 *  2. define test
 *  TEST_P(ScaleshiftConv_x3_Eltwise, CompareWithRefImpl) {
    test();
};
 *  3. INSTANTIATE_TEST_CASE_P(smoke_bfloat16_NoReshape, ScaleshiftConv_x3_Eltwise,
                        ::testing::Combine(
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(Precision::FP32),
                            ::testing::Values(SizeVector({ 1, 3, 40, 40 })),
                            ::testing::Values(SizeVector()),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ScaleshiftConv_x3_Eltwise::getTestCaseName);

 *
 * In 3rd stage do not forget bfloat16 preffix!
 */
class BasicBF16Test : public testing::WithParamInterface<basicParams>,
                      public CommonTestUtils::TestsCommon {
protected:
    virtual std::shared_ptr<ngraph::Function> createGraph(InferenceEngine::Precision netPrecision) = 0;

public:
    std::shared_ptr<ngraph::Function> fnPtr;
    std::string targetDevice;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    InferenceEngine::Precision inputPrecision, netPrecision;
    std::map<std::string, std::string> expectedPrecisions;
    float threshold = 2e-2f;  // Is enough for tensor having abs maximum values less than 1

    static std::string getTestCaseName(testing::TestParamInfo<basicParams> obj) {
        InferenceEngine::Precision inputPrecision, netPrecision;
        InferenceEngine::SizeVector inputShapes, newInputShapes;
        std::string targetDevice;
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = obj.param;

        std::ostringstream result;
        if (!newInputShapes.empty()) {
            result << "Reshape_From=" << CommonTestUtils::vec2str(inputShapes);;
            result << "_To=" << CommonTestUtils::vec2str(newInputShapes) << "_";
        } else {
            result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        }
        result << "inPRC=" << inputPrecision.name() << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    static void setNetInOutPrecision(InferenceEngine::CNNNetwork &cnnNet, InferenceEngine::Precision inPrc,
                                     InferenceEngine::Precision outPrc = InferenceEngine::Precision::UNSPECIFIED) {
        if (inPrc != InferenceEngine::Precision::UNSPECIFIED) {
            for (const auto &inputItem : cnnNet.getInputsInfo()) {
                inputItem.second->setPrecision(inPrc);
            }
        }
        if (outPrc != InferenceEngine::Precision::UNSPECIFIED) {
            for (const auto &output : cnnNet.getOutputsInfo()) {
                output.second->setPrecision(outPrc);
            }
        }
    }

    void test() {
        if (!InferenceEngine::with_cpu_x86_avx512_core()) {
            // We are enabling bf16 tests on platforms with native support bfloat16, and on platforms with AVX512 ISA
            // On platforms with AVX512 ISA but w/o native bfloat16 support computations are done via simulation mode
            GTEST_SKIP();
        }
        std::tie(inputPrecision, netPrecision, inputShapes, newInputShapes, targetDevice) = this->GetParam();
        InferenceEngine::CNNNetwork cnnNet(fnPtr);

        setNetInOutPrecision(cnnNet, inputPrecision);
        std::string inputName = cnnNet.getInputsInfo().begin()->first;
        std::string outputName = cnnNet.getOutputsInfo().begin()->first;
        auto ie = InferenceEngine::Core();
        // BF16 inference
        std::map<std::string, std::string> options;
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
        std::string inputNameFP32 = cnnNetFP32.getInputsInfo().begin()->first;
        std::string outputNameFP32 = cnnNetFP32.getOutputsInfo().begin()->first;
        setNetInOutPrecision(cnnNetFP32, inputPrecision);
        auto exec_net2 = ie.LoadNetwork(cnnNetFP32, targetDevice,
                                        { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16, InferenceEngine::PluginConfigParams::NO } });
        auto req2 = exec_net2.CreateInferRequest();


        req2.SetBlob(inputNameFP32, inBlob1);

        req2.Infer();
        auto outBlobFP32 = req2.GetBlob(outputNameFP32);
        InferenceEngine::MemoryBlob::CPtr mout2 = InferenceEngine::as<InferenceEngine::MemoryBlob>(outBlobFP32);
        ASSERT_NE(mout2, nullptr);
        auto lm2 = mout2->rmap();

        // debug to figure out the maximum value in output tensors:
        // std::cout << "Max in bfloat16 network by output " << outputName << ": " <<
        //      BFloat16Helpers::getMaxAbsValue(lm1.as<const float *>(), mout1->size()) << std::endl;
        // std::cout << "Max in fp32 network by output " << outputNameFP32 << ": " <<
        //     BFloat16Helpers::getMaxAbsValue(lm2.as<const float *>(), mout2->size()) << std::endl;
        FuncTestUtils::compareRawBuffers(lm1.as<const float *>(),
                                         lm2.as<const float *>(),
                                         mout1->size(), mout2->size(),
                                         FuncTestUtils::CompareType::ABS,
                                         threshold);
        // Stage2: verification of performance counters
        std::pair<std::string, std::string> wrongLayer =
            BFloat16Helpers::matchPerfCountPrecisionVsExpected(req1.GetPerformanceCounts(), expectedPrecisions);
        if (wrongLayer.first != std::string("")) {
            std::string layerInPerfCounts = wrongLayer.first + " " + wrongLayer.second;
            std::string layerExpected = wrongLayer.first + " " + expectedPrecisions[wrongLayer.first];
            ASSERT_EQ(layerInPerfCounts, layerExpected);
        }
        fnPtr.reset();
    }
};

}  // namespace LayerTestsDefinitions


