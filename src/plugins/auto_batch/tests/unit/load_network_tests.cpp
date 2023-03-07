// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <dimension_tracker.hpp>

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

using PluginLoadNetworkParams = std::tuple<std::map<std::string, std::string>,  // Paramters
                                           std::map<std::string, std::string>,  // Config
                                           int>;                                // Batch Size
class PluginLoadNetworkTest : public ::testing::TestWithParam<PluginLoadNetworkParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

    // Mock CPU execNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> cpuMockIExecNet;
    ov::SoPtr<IExecutableNetworkInternal> cpuMockExecNetwork;
    std::shared_ptr<NiceMock<MockIInferencePlugin>> cpuMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> cpuMockPlugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<PluginLoadNetworkParams> obj) {
        std::map<std::string, std::string> params;
        std::map<std::string, std::string> configs;
        int batch_size;
        std::tie(params, configs, batch_size) = obj.param;

        std::string res;
        for (auto& c : params) {
            res += "_" + c.first + "_" + c.second;
        }
        for (auto& c : configs) {
            res += "_" + c.first + "_" + c.second;
        }
        res += "_" + std::to_string(batch_size);
        return res;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        cpuMockIExecNet.reset();
        cpuMockExecNetwork = {};
        cpuMockPlugin = {};
    }

    void SetUp() override {
        cpuMockIExecNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        auto cpuMockIPluginPtr = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _))
            .WillByDefault(Return(cpuMockIExecNet));
        cpuMockPlugin = cpuMockIPluginPtr;
        EXPECT_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        cpuMockExecNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(cpuMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        ON_CALL(*plugin, ParseBatchDevice).WillByDefault([this](const std::string& batchDevice) {
            return plugin->AutoBatchInferencePlugin::ParseBatchDevice(batchDevice);
        });
        ON_CALL(*core, LoadNetwork(MatcherCast<const CNNNetwork&>(_), MatcherCast<const std::string&>(_), _))
            .WillByDefault(Return(cpuMockExecNetwork));
        ON_CALL(*core,
                LoadNetwork(MatcherCast<const CNNNetwork&>(_),
                            MatcherCast<const std::shared_ptr<InferenceEngine::RemoteContext>&>(_),
                            _))
            .WillByDefault(Return(cpuMockExecNetwork));
    }
};

TEST_P(PluginLoadNetworkTest, PluginLoadNetworkTestCase) {
    std::map<std::string, std::string> params;
    std::map<std::string, std::string> configs;
    int batch_size;
    std::tie(params, configs, batch_size) = this->GetParam();

    ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT"))).WillByDefault(Return(params["PERFORMANCE_HINT"]));
    ON_CALL(*core, GetMetric(_, StrEq("OPTIMAL_BATCH_SIZE"), _)).WillByDefault(Return(params["OPTIMAL_BATCH_SIZE"]));
    ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
        .WillByDefault(Return(params["PERFORMANCE_HINT_NUM_REQUESTS"]));

    ON_CALL(*core, GetMetric(_, StrEq("GPU_MEMORY_STATISTICS"), _))
        .WillByDefault([this, &params](const std::string& device, const std::string& key, const ov::AnyMap& options) {
            static int flag = 0;
            ov::Any value = params[key];
            uint64_t data = flag * value.as<uint64_t>();
            std::map<std::string, uint64_t> ret = {{"xyz", data}};
            flag = flag ? 0 : 1;
            return ret;
        });

    ON_CALL(*core, GetMetric(_, StrEq("GPU_DEVICE_TOTAL_MEM_SIZE"), _))
        .WillByDefault(Return(params["GPU_DEVICE_TOTAL_MEM_SIZE"]));

    auto graph = ngraph::builder::subgraph::makeMultiSingleConv();
    auto net = CNNNetwork(graph);
    ASSERT_NO_THROW(plugin->LoadNetworkImpl(net, {}, configs));
}

TEST_P(PluginLoadNetworkTest, PluginLoadBatchedNetworkTestCase) {
    std::map<std::string, std::string> params;
    std::map<std::string, std::string> configs;
    int batch_size;
    std::tie(params, configs, batch_size) = this->GetParam();

    ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT"))).WillByDefault(Return(params["PERFORMANCE_HINT"]));
    ON_CALL(*core, GetMetric(_, StrEq("OPTIMAL_BATCH_SIZE"), _)).WillByDefault(Return(params["OPTIMAL_BATCH_SIZE"]));
    ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
        .WillByDefault(Return(params["PERFORMANCE_HINT_NUM_REQUESTS"]));

    ON_CALL(*core, GetMetric(_, StrEq("GPU_MEMORY_STATISTICS"), _))
        .WillByDefault([this, &params](const std::string& device, const std::string& key, const ov::AnyMap& options) {
            static int flag = 0;
            ov::Any value = params[key];
            uint64_t data = flag * value.as<uint64_t>();
            std::map<std::string, uint64_t> ret = {{"xyz", data}};
            flag = flag ? 0 : 1;
            return ret;
        });

    ON_CALL(*core, GetMetric(_, StrEq("GPU_DEVICE_TOTAL_MEM_SIZE"), _))
        .WillByDefault(Return(params["GPU_DEVICE_TOTAL_MEM_SIZE"]));

    auto graph = ngraph::builder::subgraph::makeConvPoolReluNonZero({1, 1, 32, 32});
    auto batch = ov::Dimension(5);
    ov::DimensionTracker::set_label(batch, 11);
    auto p_shape = ov::PartialShape{batch, 1, 32, 32};
    graph->reshape(p_shape);
    auto net = CNNNetwork(graph);
    InferenceEngine::IExecutableNetworkInternal::Ptr execNet;
    ASSERT_NO_THROW(execNet = plugin->LoadNetworkImpl(net, {}, configs));

    ON_CALL(*cpuMockIExecNet, GetConfig(StrEq("PERFORMANCE_HINT_NUM_REQUESTS"))).WillByDefault(Return("0"));
    ON_CALL(*cpuMockIExecNet, GetMetric(StrEq("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))).WillByDefault(Return("1"));

    InferenceEngine::Parameter res;
    ASSERT_NO_THROW(res = execNet->GetMetric("OPTIMAL_NUMBER_OF_INFER_REQUESTS"));
    EXPECT_EQ(1, std::atoi(res.as<std::string>().c_str()));
}

TEST_P(PluginLoadNetworkTest, PluginLoadNetworkGetMetricTestCase) {
    std::map<std::string, std::string> params;
    std::map<std::string, std::string> configs;
    int batch_size;
    std::tie(params, configs, batch_size) = this->GetParam();

    ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT"))).WillByDefault(Return(params["PERFORMANCE_HINT"]));
    ON_CALL(*core, GetMetric(_, StrEq("OPTIMAL_BATCH_SIZE"), _)).WillByDefault(Return(params["OPTIMAL_BATCH_SIZE"]));
    ON_CALL(*core, GetConfig(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
        .WillByDefault(Return(params["PERFORMANCE_HINT_NUM_REQUESTS"]));

    ON_CALL(*core, GetMetric(_, StrEq("GPU_MEMORY_STATISTICS"), _))
        .WillByDefault([this, &params](const std::string& device, const std::string& key, const ov::AnyMap& options) {
            static int flag = 0;
            ov::Any value = params[key];
            uint64_t data = flag * value.as<uint64_t>();
            std::map<std::string, uint64_t> ret = {{"xyz", data}};
            flag = flag ? 0 : 1;
            return ret;
        });

    ON_CALL(*core, GetMetric(_, StrEq("GPU_DEVICE_TOTAL_MEM_SIZE"), _))
        .WillByDefault(Return(params["GPU_DEVICE_TOTAL_MEM_SIZE"]));

    auto graph = ngraph::builder::subgraph::makeMultiSingleConv();
    auto net = CNNNetwork(graph);
    InferenceEngine::IExecutableNetworkInternal::Ptr execNet;
    ASSERT_NO_THROW(execNet = plugin->LoadNetworkImpl(net, {}, configs));

    std::string network_name = graph.get()->get_name();
    ON_CALL(*cpuMockIExecNet, GetConfig(StrEq("PERFORMANCE_HINT_NUM_REQUESTS"))).WillByDefault(Return("0"));
    ON_CALL(*cpuMockIExecNet, GetMetric(StrEq("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))).WillByDefault(Return("1"));
    ON_CALL(*cpuMockIExecNet, GetMetric(StrEq("NETWORK_NAME"))).WillByDefault(Return(network_name.c_str()));
    ON_CALL(*cpuMockIExecNet, GetMetric(StrEq("EXECUTION_DEVICES"))).WillByDefault(Return("CPU"));

    InferenceEngine::Parameter res;
    ASSERT_NO_THROW(res = execNet->GetMetric("OPTIMAL_NUMBER_OF_INFER_REQUESTS"));
    EXPECT_EQ(batch_size, std::atoi(res.as<std::string>().c_str()));

    ASSERT_NO_THROW(res = execNet->GetMetric("NETWORK_NAME"));
    EXPECT_EQ(network_name, res.as<std::string>());

    ASSERT_NO_THROW(res = execNet->GetMetric("SUPPORTED_METRICS"));

    ASSERT_NO_THROW(res = execNet->GetMetric("EXECUTION_DEVICES"));
    EXPECT_STREQ("CPU", res.as<std::string>().c_str());

    ASSERT_ANY_THROW(execNet->GetMetric("XYZ"));
}

const std::vector<PluginLoadNetworkParams> testConfigs = {
    // Case 1: explict apply batch size by config of AUTO_BATCH_DEVICE_CONFIG
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(32)"}},
                            32},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU(32)"}},
                            32},
    // Case 2: CPU batch size is figured out by min of opt_batch_size and infReq_num
    //         If config contains "PERFORMANCE_HINT_NUM_REQUESTS" else get it from core->GetConfig
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}},
                            12},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "8"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "16"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}},
                            8},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "8"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "2"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}},
                            1},
    //PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
    //                         {"OPTIMAL_BATCH_SIZE", "32"},
    //                         {"PERFORMANCE_HINT_NUM_REQUESTS", "16"},
    //                         {"GPU_MEMORY_STATISTICS", "1024000"},
    //                         {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
    //                        {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}, {"PERFORMANCE_HINT_NUM_REQUESTS", "12"}},
    //                        12},
    //
    // Case 3: GPU batch size is figured out by
    //      1) min of opt_batch_size and infReq_num
    //      2) available_mem/one_graph_mem_footprint with power 2
    //  Final batch_size is the min of 1) and 2)
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "5000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                            4},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "40960000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                            12},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "32"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "24"},
                             {"GPU_MEMORY_STATISTICS", "1000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "18000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                            16},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "32"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "48"},
                             {"GPU_MEMORY_STATISTICS", "1000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "180000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                            32},
    // Case 4:
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "LATENCY"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(32)"}},
                            32},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         PluginLoadNetworkTest,
                         ::testing::ValuesIn(testConfigs),
                         PluginLoadNetworkTest::getTestCaseName);
