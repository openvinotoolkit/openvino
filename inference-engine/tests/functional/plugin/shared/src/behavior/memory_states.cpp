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
    std::vector<std::string> statesToQuery;
    std::tie(net, statesToQuery, targetDevice) = obj.param;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MemoryStateTest::SetUp() {
    std::tie(net, statesToQuery, deviceName) = GetParam();
}

InferenceEngine::ExecutableNetwork MemoryStateTest::PrepareNetwork() {
    net.addOutput("Memory_1");
    net.addOutput("Memory_2");
    auto ie = PluginCache::get().ie(deviceName);
    return ie->LoadNetwork(net, deviceName);
}

TEST_P(MemoryStateTest, smoke_MemoryState_QueryState) {
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
    const float new_state_val = 13.0f;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto element_count = state.GetLastState()->size();

        std::vector<float> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<float>(
            { InferenceEngine::Precision::FP32, {element_count}, InferenceEngine::C },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    for (auto&& state : executableNet.QueryState()) {
        auto lastState = state.GetLastState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();
        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int i = 0; i < last_state_size; i++) {
            EXPECT_NEAR(new_state_val, last_state_data[i], 1e-5);
        }
    }
}

TEST_P(MemoryStateTest, smoke_MemoryState_Reset) {
    auto executableNet = PrepareNetwork();
    const float new_state_val = 13.0f;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto element_count = state.GetLastState()->size();

        std::vector<float> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<float>(
            { InferenceEngine::Precision::FP32, {element_count}, InferenceEngine::C },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    executableNet.QueryState().front().Reset();

    auto states = executableNet.QueryState();
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetLastState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        if (i == 0) {
            for (int j = 0; j < last_state_size; ++j) {
                EXPECT_NEAR(0, last_state_data[j], 1e-5);
            }
        } else {
            for (int j = 0; j < last_state_size; ++j) {
                EXPECT_NEAR(13.0f, last_state_data[j], 1e-5);
            }
        }
    }
}
