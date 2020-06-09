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
    const int16_t new_state_val = 123;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto element_count = state.GetLastState()->size();

        std::vector<int16_t> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<int16_t>(
            { InferenceEngine::Precision::I16, {element_count}, InferenceEngine::C },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    for (auto&& state : executableNet.QueryState()) {
        auto lastState = state.GetLastState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<int16_t*>();
        ASSERT_FALSE(last_state_size == 0) << "State size should not be 0";

        for (int i = 0; i < last_state_size; i++) {
            ASSERT_EQ(new_state_val, last_state_data[i]);
        }
    }
}

TEST_P(MemoryStateTest, smoke_MemoryState_Reset) {
    auto executableNet = PrepareNetwork();
    const int16_t new_state_val = 123;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto element_count = state.GetLastState()->size();

        std::vector<int16_t> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<int16_t>(
            { InferenceEngine::Precision::I16, {element_count}, InferenceEngine::C },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
    }

    for (auto&& state : executableNet.QueryState()) {
        auto lastState = state.GetLastState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<int16_t*>();
        ASSERT_FALSE(last_state_size == 0) << "State size should not be 0";

        for (int i = 0; i < last_state_size; i++) {
            ASSERT_EQ(0, last_state_data[i]);
        }
    }
}
