// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <base/behavior_test_utils.hpp>
#include "behavior/infer_request/memory_states.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "blob_factory.hpp"

namespace BehaviorTestsDefinitions {
std::string InferRequestVariableStateTest::getTestCaseName(const testing::TestParamInfo<memoryStateParams> &obj) {
    std::ostringstream result;
    InferenceEngine::CNNNetwork net;
    std::string targetDevice;
    std::vector<std::string> statesToQuery;
    std::map<std::string, std::string> configuration;
    std::tie(net, statesToQuery, targetDevice, configuration) = obj.param;
    result << "targetDevice=" << targetDevice;
    if (!configuration.empty()) {
        for (auto &configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second << "_";
        }
    }
    return result.str();
}

void InferRequestVariableStateTest::SetUp() {
    std::tie(net, statesToQuery, deviceName, configuration) = GetParam();
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    IEInferRequestTestBase::SetUp();
}

InferenceEngine::CNNNetwork InferRequestVariableStateTest::getNetwork() {
    ngraph::Shape shape = {1, 200};
    ngraph::element::Type type = ngraph::element::f32;

    auto input = std::make_shared<ngraph::op::v0::Parameter>(type, shape);
    auto mem_i1 = std::make_shared<ngraph::op::v0::Constant>(type, shape, 0);
    auto mem_r1 = std::make_shared<ngraph::op::v3::ReadValue>(mem_i1, "r_1-3");
    auto mul1 = std::make_shared<ngraph::op::v1::Multiply>(mem_r1, input);

    auto mem_i2 = std::make_shared<ngraph::op::v0::Constant>(type, shape, 0);
    auto mem_r2 = std::make_shared<ngraph::op::v3::ReadValue>(mem_i2, "c_1-3");
    auto mul2 = std::make_shared<ngraph::op::v1::Multiply>(mem_r2, mul1);
    auto mem_w2 = std::make_shared<ngraph::op::v3::Assign>(mul2, "c_1-3");

    auto mem_w1 = std::make_shared<ngraph::op::v3::Assign>(mul2, "r_1-3");
    auto sigm = std::make_shared<ngraph::op::Sigmoid>(mul2);
    sigm->set_friendly_name("sigmod_state");
    mem_r1->set_friendly_name("Memory_1");
    mem_w1->add_control_dependency(mem_r1);
    sigm->add_control_dependency(mem_w1);

    mem_r2->set_friendly_name("Memory_2");
    mem_w2->add_control_dependency(mem_r2);
    sigm->add_control_dependency(mem_w2);

    auto function =
        std::make_shared<ngraph::Function>(ngraph::NodeVector{sigm}, ngraph::ParameterVector{input}, "addOutput");
    return InferenceEngine::CNNNetwork{function};
}

InferenceEngine::ExecutableNetwork InferRequestVariableStateTest::PrepareNetwork() {
    net.addOutput("Memory_1");
    net.addOutput("Memory_2");
    auto ie = PluginCache::get().ie(deviceName);
    return ie->LoadNetwork(net, deviceName, configuration);
}

TEST_P(InferRequestVariableStateTest, inferreq_smoke_VariableState_QueryState) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    auto states = inferReq.QueryState();
    ASSERT_TRUE(states.size() == 2) << "Incorrect number of VariableStates";

    for (auto &&state : states) {
        auto name = state.GetName();
        ASSERT_TRUE(std::find(statesToQuery.begin(), statesToQuery.end(), name) != statesToQuery.end())
                                    << "State " << name << "expected to be in memory states but it is not!";
    }
}

TEST_P(InferRequestVariableStateTest, inferreq_smoke_VariableState_SetState) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto &&state : inferReq.QueryState()) {
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
        delete[]new_state_data;
        state.SetState(stateBlob);
    }

    for (auto &&state : inferReq.QueryState()) {
        auto lastState = state.GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float *>();
        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";
        for (int i = 0; i < last_state_size; i++) {
            EXPECT_NEAR(new_state_val, last_state_data[i], 1e-5);
        }
    }
}

TEST_P(InferRequestVariableStateTest, inferreq_smoke_VariableState_Reset) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto &&state : inferReq.QueryState()) {
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
        delete[]new_state_data;

        state.SetState(stateBlob);
    }

    inferReq.QueryState().front().Reset();

    auto states = inferReq.QueryState();
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float *>();

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

TEST_P(InferRequestVariableStateTest, inferreq_smoke_VariableState_2infers_set) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();
    auto inferReq2 = executableNet.CreateInferRequest();

    const float new_state_val = 13.0f;
    for (auto &&state : inferReq.QueryState()) {
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
        delete[]new_state_data;
        state.SetState(stateBlob);
    }
    for (auto &&state : inferReq2.QueryState()) {
        state.Reset();
    }

    auto states = inferReq.QueryState();
    auto states2 = inferReq2.QueryState();
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float *>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(13.0f, last_state_data[j], 1e-5);
        }
    }
    for (int i = 0; i < states2.size(); ++i) {
        auto lastState = states2[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float *>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(0, last_state_data[j], 1e-5);
        }
    }
}

TEST_P(InferRequestVariableStateTest, inferreq_smoke_VariableState_2infers) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();
    auto inferReq2 = executableNet.CreateInferRequest();
    const float new_state_val = 13.0f;

    // set the input data for the network
    for (const auto &input : executableNet.GetInputsInfo()) {
        const auto &info = input.second;
        InferenceEngine::Blob::Ptr inBlob;
        inBlob = make_blob_with_precision(info->getTensorDesc());
        inBlob->allocate();
        std::memset(inBlob->buffer(), 0, inBlob->byteSize());
        inferReq.SetBlob(info->name(), inBlob);
    }

    // initial state for 2nd infer request
    for (auto &&state : inferReq2.QueryState()) {
        auto state_val = state.GetState();
        auto element_count = state_val->size();

        float *new_state_data = new float[element_count];
        for (int i = 0; i < element_count; i++) {
            new_state_data[i] = new_state_val;
        }
        auto stateBlob = make_blob_with_precision(state_val->getTensorDesc());
        stateBlob->allocate();
        std::memcpy(stateBlob->buffer(), new_state_data, element_count * sizeof(float));
        delete[]new_state_data;
        state.SetState(stateBlob);
    }

    // reset state for 1st infer request
    for (auto &&state : inferReq.QueryState()) {
        state.Reset();
    }

    inferReq.Infer();
    auto states = inferReq.QueryState();
    auto states2 = inferReq2.QueryState();
    // check the output and state of 1st request
    auto outputBlob = inferReq.GetBlob("sigmod_state");
    auto output_data = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob)->rmap().as<float*>();
    for (int i = 0; i < outputBlob->size(); i++) {
        EXPECT_NEAR(0.5f, output_data[i], 1e-5);
    }
    for (int i = 0; i < states.size(); ++i) {
        auto lastState = states[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float *>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
                EXPECT_NEAR(0.0, last_state_data[j], 1e-5);
            }
    }

    // check the output and state of 2nd request
    for (int i = 0; i < states2.size(); ++i) {
        auto lastState = states2[i].GetState();
        auto last_state_size = lastState->size();
        auto last_state_data = lastState->cbuffer().as<float *>();

        ASSERT_TRUE(last_state_size != 0) << "State size should not be 0";

        for (int j = 0; j < last_state_size; ++j) {
            EXPECT_NEAR(new_state_val, last_state_data[j], 1e-5);
        }
    }
}

TEST_P(InferRequestQueryStateExceptionTest, inferreq_smoke_QueryState_ExceptionTest) {
    auto executableNet = PrepareNetwork();
    auto inferReq = executableNet.CreateInferRequest();

    EXPECT_ANY_THROW(inferReq.QueryState());
}
} // namespace BehaviorTestsDefinitions
