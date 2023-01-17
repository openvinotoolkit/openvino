// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_auto_batch_plugin.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;
using namespace MockAutoBatchDevice;
using namespace InferenceEngine;

using PluginLoadNetworkParams = std::tuple<std::map<std::string, std::string>,  // Paramters
                                           std::map<std::string, std::string>,  // Config
                                           int>;                                // Batch Size
class PluginLoadNetworkTest : public ::testing::TestWithParam<PluginLoadNetworkParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> plugin;

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
    }

    void SetUp() override {
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        plugin->SetCore(core);

        ON_CALL(*plugin, ParseBatchDevice).WillByDefault([this](const std::string& batchDevice) {
            return plugin->AutoBatchInferencePlugin::ParseBatchDevice(batchDevice);
        });
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
            ov::Any value = params[key];
            std::map<std::string, uint64_t> ret = {{"xyz", value.as<uint64_t>()}};
            return ret;
        });
    ON_CALL(*core, GetMetric(_, StrEq("DEVICE_TOTAL_MEM_SIZE"), _))
        .WillByDefault(Return(params["DEVICE_TOTAL_MEM_SIZE"]));

    auto graph = ngraph::builder::subgraph::makeMultiSingleConv();
    auto net = CNNNetwork(graph);
    auto exeNet = plugin->LoadNetworkImpl(net, {}, configs);
}

const std::vector<PluginLoadNetworkParams> testConfigs = {
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(32)"}},
                            16},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}},
                            16},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "THROUGHPUT"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                            16},
    PluginLoadNetworkParams{{{"PERFORMANCE_HINT", "LATENCY"},
                             {"OPTIMAL_BATCH_SIZE", "16"},
                             {"PERFORMANCE_HINT_NUM_REQUESTS", "12"},
                             {"GPU_MEMORY_STATISTICS", "1024000"},
                             {"DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                            {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(32)"}},
                            16},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         PluginLoadNetworkTest,
                         ::testing::ValuesIn(testConfigs),
                         PluginLoadNetworkTest::getTestCaseName);
