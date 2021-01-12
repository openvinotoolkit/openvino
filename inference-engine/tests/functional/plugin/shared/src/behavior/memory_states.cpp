// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <common_test_utils/common_utils.hpp>
#include "behavior/memory_states.hpp"
#include "functional_test_utils/plugin_cache.hpp"

std::string VariableStateTest::getTestCaseName(const testing::TestParamInfo<memoryStateParams> &obj) {
    std::ostringstream result;
    InferenceEngine::CNNNetwork net;
    std::string targetDevice;
    std::vector<std::string> statesToQuery;
    std::tie(net, statesToQuery, targetDevice) = obj.param;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void VariableStateTest::SetUp() {
    std::tie(net, statesToQuery, deviceName) = GetParam();
}

InferenceEngine::ExecutableNetwork VariableStateTest::PrepareNetwork() {
    net.addOutput("Memory_1");
    net.addOutput("Memory_2");
    auto ie = PluginCache::get().ie(deviceName);
    return ie->LoadNetwork(net, deviceName);
}

TEST_P(VariableStateTest, smoke_VariableState_QueryState) {
    IE_SUPPRESS_DEPRECATED_START
    auto executableNet = PrepareNetwork();

    auto states = executableNet.QueryState();
    ASSERT_TRUE(states.size() == 2) << "Incorrect number of VariableStates";

    for (auto&& state : states) {
        auto name = state.GetName();
        ASSERT_TRUE(std::find(statesToQuery.begin(), statesToQuery.end(), name) != statesToQuery.end())
            << "State " << name << "expected to be in memory states but it is not!";
    }
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(VariableStateTest, smoke_VariableState_SetState) {
    IE_SUPPRESS_DEPRECATED_START
    auto executableNet = PrepareNetwork();
    const float new_state_val = 13.0f;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        std::vector<float> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<float>(
            { state_val->getTensorDesc().getPrecision(), {1, element_count}, state_val->getTensorDesc().getLayout() },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    for (auto&& state : executableNet.QueryState()) {
        auto lastState = state.GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();
        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int i = 0; i < last_state_size; i++) {
            EXPECT_NEAR(new_state_val, last_state_data[i], 1e-5);
        }
    }
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(VariableStateTest, smoke_VariableState_Reset) {
    IE_SUPPRESS_DEPRECATED_START
    auto executableNet = PrepareNetwork();
    const float new_state_val = 13.0f;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        std::vector<float> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<float>(
            { state_val->getTensorDesc().getPrecision(), {1, element_count}, state_val->getTensorDesc().getLayout() },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    executableNet.QueryState().front().Reset();

    auto states = executableNet.QueryState();
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetState();
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
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(VariableStateTest, inferreq_smoke_VariableState_QueryState) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    auto states = inferReq.QueryState();
    ASSERT_TRUE(states.size() == 2) << "Incorrect number of VariableStates";

    for (auto&& state : states) {
        auto name = state.GetName();
        ASSERT_TRUE(std::find(statesToQuery.begin(), statesToQuery.end(), name) != statesToQuery.end())
            << "State " << name << "expected to be in memory states but it is not!";
    }
}

TEST_P(VariableStateTest, inferreq_smoke_VariableState_SetState) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto&& state : inferReq.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        std::vector<float> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<float>(
            { state_val->getTensorDesc().getPrecision(), {1, element_count}, state_val->getTensorDesc().getLayout() },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    for (auto&& state : inferReq.QueryState()) {
        auto lastState = state.GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();
        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int i = 0; i < last_state_size; i++) {
            EXPECT_NEAR(new_state_val, last_state_data[i], 1e-5);
        }
    }
}

TEST_P(VariableStateTest, inferreq_smoke_VariableState_Reset) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto&& state : inferReq.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        std::vector<float> new_state_data(element_count, new_state_val);
        auto stateBlob = InferenceEngine::make_shared_blob<float>(
            { state_val->getTensorDesc().getPrecision(), {1, element_count}, state_val->getTensorDesc().getLayout() },
            new_state_data.data(), new_state_data.size());

        state.SetState(stateBlob);
    }

    inferReq.QueryState().front().Reset();

    auto states = inferReq.QueryState();
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetState();
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
