// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <common_test_utils/common_utils.hpp>
#include "behavior/memory_states.hpp"
#include "functional_test_utils/plugin_cache.hpp"

std::string MemoryStateTest::getTestCaseName(const testing::TestParamInfo<memoryStateParams> &obj) {
    std::ostringstream result;
    InferenceEngine::CNNNetwork net;
    std::string targetDevice;
    std::tie(net, targetDevice) = obj.param;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MemoryStateTest::SetUp() {
    std::tie(net, deviceName) = GetParam();
}

InferenceEngine::ExecutableNetwork MemoryStateTest::PrepareNetwork() {
    net.addOutput("Memory_1");
    net.addOutput("Memory_2");
    auto ie = PluginCache::get().ie(deviceName);
    return ie->LoadNetwork(net, deviceName);
}

TEST_P(MemoryStateTest, smoke_MemoryState_QueryState) {
    std::vector<std::string> statesToQuery = {"c_1-3", "r_1-3"};
    auto executableNet = PrepareNetwork();

    auto states = executableNet.QueryState();
    ASSERT_TRUE(states.size() == 2) << "Incorrect number of MemoryStates";

    for (auto&& state : states) {
        auto name = state.GetName();
        ASSERT_TRUE(std::find(statesToQuery.begin(), statesToQuery.end(), name) != statesToQuery.end())
            << "State " << name << "expected to be in memory states but it is not!";
    }
}

TEST_P(MemoryStateTest, smoke_MemoryState_SetState) {
    auto executableNet = PrepareNetwork();

    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
    }

    float data[] = { 123, 124 };
    auto stateBlob = InferenceEngine::make_shared_blob<float>(
        { InferenceEngine::Precision::FP32, {2}, InferenceEngine::C },
        data, sizeof(data) / sizeof(*data));
    executableNet.QueryState().front().SetState(stateBlob);


    for (auto&& state : executableNet.QueryState()) {
        try {
            auto lastState = state.GetLastState();
        } catch (InferenceEngine::details::InferenceEngineException ex) {
            std::string msg = ex.what();
            ASSERT_TRUE(msg.find("GetLastState method is not yet implemented for GNAMemoryState") != std::string::npos) << ex.what();
        }
    }
}
