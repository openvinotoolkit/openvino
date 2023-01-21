// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "mock_auto_batch_plugin.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;
using namespace MockAutoBatchPlugin;
using namespace MockAutoBatchDevice;
using namespace InferenceEngine;

using CreateInferRequestTestParams = std::tuple<int,  // batch_size
                                                int>; // inferReq number
class CreateInferRequestTest : public ::testing::TestWithParam<CreateInferRequestTestParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

    // Mock execNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> mockIExecNet;
    ov::SoPtr<IExecutableNetworkInternal> mockExecNetwork;
    std::shared_ptr<NiceMock<MockIInferencePlugin>> mockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> mockPlugin;
    ov::SoPtr<IExecutableNetworkInternal> batchedExecNetwork;

    std::shared_ptr<AutoBatchExecutableNetwork> actualExecNet;
    std::vector<std::shared_ptr<NiceMock<MockIInferRequestInternal>>> inferRequestVec;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CreateInferRequestTestParams> obj) {
        int batch_size;
        int infer_num;
        std::tie(batch_size, infer_num) = obj.param;

        std::string res;
        res = "batch_size_" + std::to_string(batch_size);
        res += "_infer_num_" + std::to_string(infer_num);
        return res;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        mockIExecNet.reset();
        mockExecNetwork = {};
        batchedExecNetwork = {};
        mockPlugin = {};
        actualExecNet.reset();
        inferRequestVec.clear();
    }

    void SetUp() override {
        mockIExecNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        mockIPlugin = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*mockIPlugin, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockIExecNet));
        mockPlugin = mockIPlugin;
        mockExecNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(mockPlugin->LoadNetwork(CNNNetwork{}, {}), {});
        batchedExecNetwork = {};

        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        // Create inferRequest
        ON_CALL(*mockIExecNet.get(), CreateInferRequest()).WillByDefault([this]() {
            auto inferReq = std::make_shared<NiceMock<MockIInferRequestInternal>>();
            inferRequestVec.push_back(inferReq);
            return inferReq;
        });
    }

    AutoBatchExecutableNetwork::Ptr createAutoBatchExecutableNetwork(int batch_size) {
        DeviceInformation metaDevice = {"CPU", {}, batch_size};
        std::unordered_map<std::string, InferenceEngine::Parameter> config = {{CONFIG_KEY(AUTO_BATCH_TIMEOUT), "200"}};
        std::set<std::string> batched_inputs = {"Parameter_0"};
        std::set<std::string> batched_outputs = {"Convolution_20"};

        if (batch_size > 1)
            batchedExecNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(mockPlugin->LoadNetwork(CNNNetwork{}, {}), {});
        return std::make_shared<AutoBatchExecutableNetwork>(batchedExecNetwork,
                                                            mockExecNetwork,
                                                            metaDevice,
                                                            config,
                                                            batched_inputs,
                                                            batched_outputs);
    }
};

TEST_P(CreateInferRequestTest, CreateInferRequestTestCases) {
    int batch_size;
    int infer_num;
    std::tie(batch_size, infer_num) = this->GetParam();

    actualExecNet = createAutoBatchExecutableNetwork(batch_size);
    std::vector<InferenceEngine::IInferRequestInternal::Ptr> inferReqs;
    InferenceEngine::IInferRequestInternal::Ptr inferReq;
    for (int i = 0; i < infer_num; i++) {
        EXPECT_NO_THROW(inferReq = actualExecNet->CreateInferRequest());
        EXPECT_NE(inferReq, nullptr);
        inferReqs.push_back(inferReq);
    }
    inferReqs.clear();
}

const std::vector<int> requests_num{1, 8, 16, 64};
const std::vector<int> batch_size{1, 8, 16, 32, 128, 256};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         CreateInferRequestTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(batch_size),
                            ::testing::ValuesIn(requests_num)),
                         CreateInferRequestTest::getTestCaseName);