// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <tests_common_func.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include <functional_test_utils/blob_utils.hpp>

using namespace ::testing;
using namespace InferenceEngine;

class smoke_PropertyTest : public TestsCommon, public TestsCommonFunc{};

TEST_F(smoke_PropertyTest, onSplitConvConcat) {
    auto fnPtr = ngraph::builder::subgraph::makeSplitConvConcat({1, 4, 100, 100});

    CNNNetwork net(fnPtr);
    auto ieCore = PluginCache::get().ie();
    InferenceEngine::ExecutableNetwork exeNet = ieCore->LoadNetwork(net, CommonTestUtils::DEVICE_CPU);
    InferenceEngine::InferRequest inferRequest0 = exeNet.CreateInferRequest();

    auto blob0 = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    inferRequest0.SetBlob(net.getInputsInfo().begin()->first, blob0);
    inferRequest0.Infer();
    float* outRawData = inferRequest0.GetBlob(net.getOutputsInfo().begin()->first)->cbuffer().as<float*>();


    exeNet = ieCore->LoadNetwork(net, CommonTestUtils::DEVICE_CPU,
            {{PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, PluginConfigParams::CPU_THROUGHPUT_AUTO}});
    InferenceEngine::InferRequest inferRequest1 = exeNet.CreateInferRequest();

    auto blob1 = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    inferRequest1.SetBlob(net.getInputsInfo().begin()->first, blob1);
    inferRequest1.Infer();
    float* outRawDataWithConfig = inferRequest1.GetBlob(net.getOutputsInfo().begin()->first)->cbuffer().as<float*>();

    float thr1, thr2;
    FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32, thr1, thr2);

    size_t outElementsCount = std::accumulate(begin(fnPtr->get_output_shape(0)), end(fnPtr->get_output_shape(0)), 1,
                                              std::multiplies<size_t>());

    FuncTestUtils::compareRawBuffers(outRawData, outRawDataWithConfig, outElementsCount, outElementsCount,
                                                     FuncTestUtils::CompareType::ABS_AND_REL,
                                                     thr1, thr2);
}
