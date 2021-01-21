// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <base/behavior_test_utils.hpp>
#include <common_test_utils/common_utils.hpp>
#include "behavior/memory_states.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "blob_factory.hpp"

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
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    IE_SUPPRESS_DEPRECATED_START
    auto executableNet = PrepareNetwork();
    const float new_state_val = 13.0f;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        float *new_state_data = new float[element_count];
        for (int i = 0; i < element_count; i++) {
            new_state_data[i] = new_state_val;
        }
        auto stateBlob = make_blob_with_precision(state_val->getTensorDesc());
        stateBlob->allocate();
        std::memcpy(stateBlob->buffer(), new_state_data, element_count * sizeof(float));
        delete []new_state_data;
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
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    IE_SUPPRESS_DEPRECATED_START
    auto executableNet = PrepareNetwork();
    const float new_state_val = 13.0f;
    for (auto&& state : executableNet.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        float *new_state_data = new float[element_count];
        for (int i = 0; i < element_count; i++) {
            new_state_data[i] = new_state_val;
        }
        auto stateBlob = make_blob_with_precision(state_val->getTensorDesc());
        stateBlob->allocate();
        std::memcpy(stateBlob->buffer(), new_state_data, element_count * sizeof(float));
        delete []new_state_data;

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
                EXPECT_NEAR(new_state_val, last_state_data[j], 1e-5);
            }
        }
    }
    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(VariableStateTest, inferreq_smoke_VariableState_QueryState) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto&& state : inferReq.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        float *new_state_data = new float[element_count];
        for (int i = 0; i < element_count; i++) {
            new_state_data[i] = new_state_val;
        }
        auto stateBlob = make_blob_with_precision(state_val->getTensorDesc());
        stateBlob->allocate();
        std::memcpy(stateBlob->buffer(), new_state_data, element_count * sizeof(float));
        delete []new_state_data;
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
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto&& state : inferReq.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        float *new_state_data = new float[element_count];
        for (int i = 0; i < element_count; i++) {
            new_state_data[i] = new_state_val;
        }
        auto stateBlob = make_blob_with_precision(state_val->getTensorDesc());
        stateBlob->allocate();
        std::memcpy(stateBlob->buffer(), new_state_data, element_count * sizeof(float));
        delete []new_state_data;

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
                EXPECT_NEAR(new_state_val, last_state_data[j], 1e-5);
            }
        }
    }
}

TEST_P(VariableStateTest, inferreq_smoke_VariableState_2infers_set) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();
    auto inferReq2 = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto&& state : inferReq.QueryState()) {
        state.Reset();
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        float *new_state_data = new float[element_count];
        for (int i = 0; i < element_count; i++) {
            new_state_data[i] = new_state_val;
        }
        auto stateBlob = make_blob_with_precision(state_val->getTensorDesc());
        stateBlob->allocate();
        std::memcpy(stateBlob->buffer(), new_state_data, element_count * sizeof(float));
        delete []new_state_data;
        state.SetState(stateBlob);
    }
    for (auto&& state : inferReq2.QueryState()) {
        state.Reset();
    }

    auto states = inferReq.QueryState();
    auto states2 = inferReq2.QueryState();
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(13.0f, last_state_data[j], 1e-5);
        }
    }
    for (int i = 0; i < states2.size(); ++i) {
        auto lastState = states2[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(0, last_state_data[j], 1e-5);
        }
    }
}

TEST_P(VariableStateTest, inferreq_smoke_VariableState_2infers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();
    auto inferReq2 = executableNet.CreateInferRequest();

    for (const auto &input : executableNet.GetInputsInfo()) {
        const auto &info = input.second;
        InferenceEngine::Blob::Ptr inBlob;
        inBlob = make_blob_with_precision(info->getTensorDesc());
        inBlob->allocate();
        std::memset(inBlob->buffer(), 0, inBlob->byteSize());
        inferReq.SetBlob(info->name(), inBlob);
    }

    for (auto&& state : inferReq.QueryState()) {
        state.Reset();
    }
    for (auto&& state : inferReq2.QueryState()) {
        state.Reset();
    }

    inferReq.Infer();

    auto states = inferReq.QueryState();
    auto states2 = inferReq2.QueryState();
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        if (i == 0) {
            for (int j = 0; j < last_state_size; ++j) {
                EXPECT_NEAR(0.5f, last_state_data[j], 1e-3);
            }
        } else {
            for (int j = 0; j < last_state_size; ++j) {
                EXPECT_NEAR(0.0f, last_state_data[j], 1e-5);
            }
        }
    }
    for (int i = 0; i < states2.size(); ++i) {
        auto lastState = states2[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float*>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(0.0f, last_state_data[j], 1e-5);
        }
    }
}
