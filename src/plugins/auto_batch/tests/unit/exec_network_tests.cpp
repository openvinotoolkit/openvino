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

using ExecNetworkParams = std::tuple<std::string,  // Key name
                                     int,          // GetMetric(0) or GetConfig(1) or SetConfig(3)
                                     bool>;        // Throw exception
class ExecNetworkTest : public ::testing::TestWithParam<ExecNetworkParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

    // Mock execNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> mockIExecNet;
    ov::SoPtr<IExecutableNetworkInternal> mockExecNetwork;
    std::shared_ptr<NiceMock<MockIInferencePlugin>> mockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> mockPlugin;

    InferenceEngine::IExecutableNetworkInternal::Ptr actualExecNet;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ExecNetworkParams> obj) {
        std::string name;
        bool throw_exception;
        int action;
        std::tie(name, action, throw_exception) = obj.param;

        std::string res;
        switch (action) {
        case 0:
            res += "GetMetric_" + name;
            break;
        case 1:
            res += "GetConfig_" + name;
            break;
        case 3:
            res += "SetConfig_" + name;
            break;
        default:
            res += "error_" + name;
        }

        if (throw_exception)
            res += "throw";

        return res;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        mockIExecNet.reset();
        mockExecNetwork = {};
        mockPlugin = {};
        actualExecNet.reset();
    }

    void SetUp() override {
        mockIExecNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        auto mockIPluginPtr = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).WillByDefault(Return(mockIExecNet));
        mockPlugin = mockIPluginPtr;
        EXPECT_CALL(*mockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        mockExecNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(mockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        ON_CALL(*plugin, ParseBatchDevice).WillByDefault([this](const std::string& batchDevice) {
            return plugin->AutoBatchInferencePlugin::ParseBatchDevice(batchDevice);
        });
        ON_CALL(*core, LoadNetwork(MatcherCast<const CNNNetwork&>(_), MatcherCast<const std::string&>(_), _))
            .WillByDefault(Return(mockExecNetwork));
        ON_CALL(*core,
                LoadNetwork(MatcherCast<const CNNNetwork&>(_),
                            MatcherCast<const std::shared_ptr<InferenceEngine::RemoteContext>&>(_),
                            _))
            .WillByDefault(Return(mockExecNetwork));
        ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT"))).WillByDefault(Return("THROUGHPUT"));
        ON_CALL(*core, GetMetric(_, StrEq("OPTIMAL_BATCH_SIZE"), _)).WillByDefault(Return("16"));
        ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS"))).WillByDefault(Return("12"));
        ON_CALL(*core, GetMetric(_, StrEq("GPU_MEMORY_STATISTICS"), _))
            .WillByDefault([](const std::string& device, const std::string& key, const ov::AnyMap& options) {
                std::map<std::string, uint64_t> ret = {{"xyz", 1024}};
                return ret;
            });
        ON_CALL(*core, GetMetric(_, StrEq("GPU_DEVICE_TOTAL_MEM_SIZE"), _)).WillByDefault(Return("10240"));
        auto graph = ngraph::builder::subgraph::makeMultiSingleConv();
        auto net = CNNNetwork(graph);

        const std::map<std::string, std::string> configs = {{"AUTO_BATCH_TIMEOUT", "200"},
                                                            {"AUTO_BATCH_DEVICE_CONFIG", "CPU(16)"}};
        ASSERT_NO_THROW(actualExecNet = plugin->LoadNetworkImpl(net, {}, configs));

        ON_CALL(*mockIExecNet, GetConfig(StrEq("PERFORMANCE_HINT_NUM_REQUESTS"))).WillByDefault(Return("0"));
        ON_CALL(*mockIExecNet, GetMetric(StrEq("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))).WillByDefault(Return("12"));
        ON_CALL(*mockIExecNet, GetMetric(StrEq("NETWORK_NAME"))).WillByDefault(Return("network_name"));
        ON_CALL(*mockIExecNet, GetMetric(StrEq("EXECUTION_DEVICES"))).WillByDefault(Return("CPU"));
        ON_CALL(*mockIExecNet, GetMetric(StrEq("SUPPORTED_CONFIG_KEYS"))).WillByDefault(Return("CPU"));
        ON_CALL(*mockIExecNet, GetMetric(StrEq("SUPPORTED_CONFIG_KEYS"))).WillByDefault([](const std::string& name) {
            std::vector<std::string> res_config;
            res_config.emplace_back("CACHE_DIR");
            res_config.emplace_back("OPTIMAL_BATCH_SIZE");
            return res_config;
        });
        ON_CALL(*mockIExecNet, GetConfig(StrEq("CACHE_DIR"))).WillByDefault(Return("./abc"));
        ON_CALL(*mockIExecNet, GetConfig(StrEq("OPTIMAL_BATCH_SIZE"))).WillByDefault(Return("16"));
    }
};

TEST_P(ExecNetworkTest, ExecNetworkGetConfigMetricTestCase) {
    std::string name;
    bool throw_exception;
    int action;
    std::tie(name, action, throw_exception) = this->GetParam();

    std::map<std::string, InferenceEngine::Parameter> config;

    switch (action) {
    case 0: {
        if (throw_exception)
            ASSERT_ANY_THROW(actualExecNet->GetMetric(name));
        else
            ASSERT_NO_THROW(actualExecNet->GetMetric(name));
        break;
    }
    case 1: {
        if (throw_exception)
            ASSERT_ANY_THROW(actualExecNet->GetConfig(name));
        else
            ASSERT_NO_THROW(actualExecNet->GetConfig(name));
        break;
    }
    case 3: {
        config[name] = InferenceEngine::Parameter(100);
        if (throw_exception)
            ASSERT_ANY_THROW(actualExecNet->SetConfig(config));
        else
            ASSERT_NO_THROW(actualExecNet->SetConfig(config));
        break;
    }
    default:
        break;
    }
}

const std::vector<ExecNetworkParams> testConfigs = {
        // Metric
        ExecNetworkParams{METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS), 0, false},
        ExecNetworkParams{METRIC_KEY(NETWORK_NAME), 0, false},
        ExecNetworkParams{METRIC_KEY(SUPPORTED_METRICS), 0, false},
        ExecNetworkParams{METRIC_KEY(SUPPORTED_CONFIG_KEYS), 0, false},
        ExecNetworkParams{ov::execution_devices.name(), 0, false},
        // Config in autobatch
        ExecNetworkParams{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), 1, false},
        ExecNetworkParams{CONFIG_KEY(AUTO_BATCH_TIMEOUT), 1, false},
        ExecNetworkParams{CONFIG_KEY(CACHE_DIR), 1, false},
        // Config in dependent plugin
        ExecNetworkParams{"OPTIMAL_BATCH_SIZE", 1, false},
        // Incorrect Metric
        ExecNetworkParams{"INCORRECT_METRIC", 0, true},
        // Incorrect config
        ExecNetworkParams{"INCORRECT_CONFIG", 1, true},
        // Set Config
        ExecNetworkParams{CONFIG_KEY(AUTO_BATCH_TIMEOUT), 2, false},
        ExecNetworkParams{"INCORRECT_CONFIG", 2, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         ExecNetworkTest,
                         ::testing::ValuesIn(testConfigs),
                         ExecNetworkTest::getTestCaseName);