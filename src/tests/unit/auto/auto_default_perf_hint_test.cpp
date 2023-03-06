// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common_test_utils/test_constants.hpp>
#include <ie_core.hpp>
#include <ie_metric_helpers.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <openvino/runtime/core.hpp>

#include "mock_common.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "plugin/mock_load_network_properties.hpp"
#include "so_ptr.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"

using ::testing::_;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::Throw;
using Config = std::map<std::string, std::string>;

// define a matcher to check if perf hint expects
MATCHER_P(ComparePerfHint, perfHint, "Check if perf hint expects.") {
    std::string arg_perfHint = "No PERFORMANCE_HINT";
    auto itor = arg.find(PluginConfigParams::KEY_PERFORMANCE_HINT);
    if (itor != arg.end()) {
        arg_perfHint = itor->second;
    }

    return perfHint == arg_perfHint;
}
using namespace MockMultiDevice;

using ConfigParams = std::tuple<std::string,               // virtual device name to load network
                                std::vector<std::string>,  // hardware device name to expect loading network on
                                Config>;                   // secondary property setting to device

static std::vector<ConfigParams> testConfigs;

class AutoDefaultPerfHintTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>> plugin;
    InferenceEngine::CNNNetwork simpleCnnNetwork;
    // mock cpu exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> cpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal> cpuMockExeNetwork;
    NiceMock<MockIInferencePlugin>* cpuMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> cpuMockPlugin;

    // mock gpu exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> gpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal> gpuMockExeNetwork;
    NiceMock<MockIInferencePlugin>* gpuMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> gpuMockPlugin;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>> inferReqInternal;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string deviceName;
        std::vector<std::string> targetDevices;
        Config deviceConfigs;
        std::tie(deviceName, targetDevices, deviceConfigs) = obj.param;
        std::ostringstream result;
        result << deviceName;
        result << "_loadnetwork_to_";
        for (auto& device : targetDevices) {
            result << device << "_";
        }
        auto cpuConfig = deviceConfigs.find("CPU");
        auto gpuConfig = deviceConfigs.find("GPU");
        auto priority = deviceConfigs.find("MULTI_DEVICE_PRIORITIES");
        result << "properties_";
        if (cpuConfig != deviceConfigs.end())
            result << "CPU_" << cpuConfig->second << "_";
        if (gpuConfig != deviceConfigs.end())
            result << "GPU_" << gpuConfig->second << "_";
        if (priority != deviceConfigs.end())
            result << "priority_" << priority->second;
        return result.str();
    }

    static std::vector<ConfigParams> CreateNumStreamsAndDefaultPerfHintTestConfigs() {
        testConfigs.clear();
        testConfigs.push_back(
            ConfigParams{"AUTO", {"CPU"}, {{"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: get default_hint:lantency
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU"},
                         {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get default_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"CPU", "GPU"},
            {{"CPU", "NUM_STREAMS 3"},
             {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});  // CPU: as helper, get default_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"CPU", "GPU"},
            {{"GPU", "NUM_STREAMS 3"},
             {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});  // CPU: as helper, get default_hint:lantency GPU:no perf_hint
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU"},
                         {{"CPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"AUTO", {"GPU"}, {{"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: get default_hint:lantency
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"GPU"},
                         {{"GPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: no perf_hint

        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: get default_hint:tput
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"CPU", "NUM_STREAMS 3"},
                          {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint  GPU: get default_hint:tput
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"GPU", "NUM_STREAMS 3"},
                          {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"MULTI:CPU,GPU",
                         {"CPU", "GPU"},
                         {{"CPU", "NUM_STREAMS 3"},
                          {"GPU", "NUM_STREAMS 3"},
                          {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint  GPU: no perf_hint
        return testConfigs;
    }

    static std::vector<ConfigParams> CreatePerfHintAndDefaultPerfHintTestConfigs() {
        testConfigs.clear();
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"CPU"},
            {{"CPU", "PERFORMANCE_HINT THROUGHPUT"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: get perf_hint:tput
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"CPU", "PERFORMANCE_HINT THROUGHPUT"},
                          {"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"CPU", "PERFORMANCE_HINT THROUGHPUT"},
                          {"GPU", "PERFORMANCE_HINT THROUGHPUT"},
                          {"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:get perf_hint:tput
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"CPU", "PERFORMANCE_HINT THROUGHPUT"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get perf_hint:tput
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"GPU"},
            {{"GPU", "PERFORMANCE_HINT THROUGHPUT"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: get perf_hint:tput

        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"CPU", "PERFORMANCE_HINT LATENCY"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get perf_hint:latency  GPU: get default_hint:tput
        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"GPU", "PERFORMANCE_HINT LATENCY"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: get perf_hint:latency
        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"CPU", "PERFORMANCE_HINT LATENCY"},
             {"GPU", "PERFORMANCE_HINT LATENCY"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get perf_hint:lantency  GPU: get perf_hint:lantency
        return testConfigs;
    }

    static std::vector<ConfigParams> CreateSecPropAndDefaultPerfHintTestConfigs() {
        testConfigs.clear();
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"CPU"},
            {{"CPU", "ALLOW_AUTO_BATCHING TRUE"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"CPU", "ALLOW_AUTO_BATCHING TRUE"},
                          {"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:get default_hint:lantency
        testConfigs.push_back(
            ConfigParams{"AUTO",
                         {"CPU", "GPU"},
                         {{"CPU", "ALLOW_AUTO_BATCHING TRUE"},
                          {"GPU", "ALLOW_AUTO_BATCHING TRUE"},
                          {"MULTI_DEVICE_PRIORITIES",
                           "GPU,CPU"}}});  // CPU: as helper, get perf_hint:lantency GPU:no perf_hint
        testConfigs.push_back(ConfigParams{"AUTO",
                                           {"CPU"},
                                           {{"CPU", "ALLOW_AUTO_BATCHING FALSE"},
                                            {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint
        testConfigs.push_back(ConfigParams{
            "AUTO",
            {"GPU"},
            {{"GPU", "ALLOW_AUTO_BATCHING FALSE"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});  // GPU: no perf_hint

        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"CPU", "ALLOW_AUTO_BATCHING FALSE"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint  GPU: get default_hint:tput
        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"GPU", "ALLOW_AUTO_BATCHING FALSE"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: get default_hint:tput  GPU: no perf_hint
        testConfigs.push_back(ConfigParams{
            "MULTI:CPU,GPU",
            {"CPU", "GPU"},
            {{"CPU", "ALLOW_AUTO_BATCHING TRUE"},
             {"GPU", "ALLOW_AUTO_BATCHING FALSE"},
             {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});  // CPU: no perf_hint GPU: no perf_hint
        return testConfigs;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
        // prepare cpuMockExeNetwork
        cpuMockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        auto cpuMockIPluginPtr = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _))
            .WillByDefault(Return(cpuMockIExeNet));
        cpuMockPlugin = cpuMockIPluginPtr;
        // remove annoying ON CALL message
        EXPECT_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        cpuMockExeNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(cpuMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

        // prepare gpuMockExeNetwork
        gpuMockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
        auto gpuMockIPluginPtr = std::make_shared<NiceMock<MockIInferencePlugin>>();
        ON_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _))
            .WillByDefault(Return(gpuMockIExeNet));
        gpuMockPlugin = gpuMockIPluginPtr;
        // remove annoying ON CALL message
        EXPECT_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        gpuMockExeNetwork = ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(gpuMockPlugin->LoadNetwork(CNNNetwork{}, {}), {});

        // prepare mockicore and cnnNetwork for loading
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        auto* origin_plugin = new NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>();
        plugin = std::shared_ptr<NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>>(origin_plugin);
        // replace core with mock Icore
        plugin->SetCore(core);
        inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        ON_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
        ON_CALL(*gpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));

        ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));
        ON_CALL(*gpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));

        std::vector<std::string> availableDevs = {"CPU", "GPU"};
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));

        std::vector<std::string> metrics = {METRIC_KEY(SUPPORTED_CONFIG_KEYS)};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _)).WillByDefault(Return(metrics));

        std::vector<std::string> configKeys = {"SUPPORTED_CONFIG_KEYS", "NUM_STREAMS"};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(configKeys));

        ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name()))).WillByDefault(Return(12));

        ON_CALL(*plugin, ParseMetaDevices)
            .WillByDefault(
                [this](const std::string& priorityDevices, const std::map<std::string, std::string>& config) {
                    return plugin->MultiDeviceInferencePlugin::ParseMetaDevices(priorityDevices, config);
                });

        ON_CALL(*plugin, SelectDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int priority) {
                return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, priority);
            });

        ON_CALL(*plugin, GetValidDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });

        ON_CALL(*core, GetSupportedConfig)
            .WillByDefault([](const std::string& device, const std::map<std::string, std::string>& fullConfigs) {
                auto item = fullConfigs.find(device);
                Config deviceConfigs;
                if (item != fullConfigs.end()) {
                    std::stringstream strConfigs(item->second);
                    ov::util::Read<Config>{}(strConfigs, deviceConfigs);
                }
                return deviceConfigs;
            });

        ON_CALL(*plugin, GetDeviceList).WillByDefault([this](const std::map<std::string, std::string>& config) {
            return plugin->MultiDeviceInferencePlugin::GetDeviceList(config);
        });
        ON_CALL(*plugin, SelectDevice)
            .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices,
                                  const std::string& netPrecision,
                                  unsigned int Priority) {
                return plugin->MultiDeviceInferencePlugin::SelectDevice(metaDevices, netPrecision, Priority);
            });
        std::vector<std::string> cpuCability{"FP32", "FP16", "INT8", "BIN"};
        std::vector<std::string> gpuCability{"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"};
        ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_CPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(cpuCability));
        ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(Return(gpuCability));

        ON_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(StrEq("CPU")),
                            ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .WillByDefault(Return(cpuMockExeNetwork));

        ON_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(StrEq("GPU")),
                            ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .WillByDefault(Return(gpuMockExeNetwork));

        std::shared_ptr<ngraph::Function> simpleNetwork = ngraph::builder::subgraph::makeSingleConv();
        ASSERT_NO_THROW(simpleCnnNetwork = InferenceEngine::CNNNetwork(simpleNetwork));
    }
};

using NumStreamsAndDefaultPerfHintMockTest = AutoDefaultPerfHintTest;
using PerHintAndDefaultPerfHintMockTest = AutoDefaultPerfHintTest;
using SecPropAndDefaultPerfHintMockTest = AutoDefaultPerfHintTest;

TEST_P(NumStreamsAndDefaultPerfHintMockTest, NumStreamsAndDefaultPerfHintTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    Config config;
    bool bIsAuto = false;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos) {
        bIsAuto = true;
        plugin->SetName("AUTO");
    }

    if (device.find("MULTI") != std::string::npos)
        plugin->SetName("MULTI");

    for (auto& deviceName : targetDevices) {
        bool isCPUHelper = false;
        if (deviceName.find("CPU") != std::string::npos && bIsAuto) {
            auto item = config.find("MULTI_DEVICE_PRIORITIES");
            if (item != config.end() && item->second.find("GPU,CPU") != std::string::npos) {
                isCPUHelper = true;
            }
        }
        std::string HW_PerfHint;
        if (isCPUHelper) {
            // if it is CPU Helper, CPU should keep perf hint to LATENCY.
            HW_PerfHint = "LATENCY";
        } else {
            // HW default perf_hint
            HW_PerfHint = bIsAuto ? "LATENCY" : "THROUGHPUT";
        }
        auto item = config.find(deviceName);
        Config deviceConfigs;
        if (item != config.end()) {
            std::stringstream strConfigs(item->second);
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<Config>{}(strConfigs, deviceConfigs);
        }
        bool bNumStreams = deviceConfigs.find(ov::num_streams.name()) != deviceConfigs.end() ? true : false;
        if (bNumStreams && !isCPUHelper) {
            // do not pass default perf_hint to HW
            HW_PerfHint = "No PERFORMANCE_HINT";
        }
        EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(deviceName),
                        ::testing::Matcher<const std::map<std::string, std::string>&>(ComparePerfHint(HW_PerfHint))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, config));
}

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiMock_NumStreamsAndDefaultPerfHintToHWTest,
                         NumStreamsAndDefaultPerfHintMockTest,
                         ::testing::ValuesIn(NumStreamsAndDefaultPerfHintMockTest::CreateNumStreamsAndDefaultPerfHintTestConfigs()),
                         NumStreamsAndDefaultPerfHintMockTest::getTestCaseName);

TEST_P(PerHintAndDefaultPerfHintMockTest, PerfHintAndDefaultPerfHintTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    Config config;
    bool bIsAuto = false;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos) {
        bIsAuto = true;
        plugin->SetName("AUTO");
    }

    if (device.find("MULTI") != std::string::npos)
        plugin->SetName("MULTI");

    for (auto& deviceName : targetDevices) {
        bool isCPUHelper = false;
        if (deviceName.find("CPU") != std::string::npos && bIsAuto) {
            auto item = config.find("MULTI_DEVICE_PRIORITIES");
            if (item != config.end() && item->second.find("GPU,CPU") != std::string::npos) {
                isCPUHelper = true;
            }
        }
        std::string HW_PerfHint;
        if (isCPUHelper) {
            // if it is CPU Helper, CPU should keep perf hint to LATENCY.
            HW_PerfHint = "LATENCY";
        } else {
            // HW default perf_hint
            HW_PerfHint = bIsAuto ? "LATENCY" : "THROUGHPUT";
        }
        auto item = config.find(deviceName);
        Config deviceConfigs;
        if (item != config.end()) {
            std::stringstream strConfigs(item->second);
            // Parse the device properties to common property into deviceConfigs.
            ov::util::Read<Config>{}(strConfigs, deviceConfigs);
        }
        auto itor = deviceConfigs.find(PluginConfigParams::KEY_PERFORMANCE_HINT);
        if (itor != deviceConfigs.end() && !isCPUHelper) {
            HW_PerfHint = itor->second;
        }
        EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(deviceName),
                        ::testing::Matcher<const std::map<std::string, std::string>&>(ComparePerfHint(HW_PerfHint))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, config));
}

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiMock_PerHintAndDefaultPerfHintToHWTest,
                         PerHintAndDefaultPerfHintMockTest,
                         ::testing::ValuesIn(PerHintAndDefaultPerfHintMockTest::CreatePerfHintAndDefaultPerfHintTestConfigs()),
                         PerHintAndDefaultPerfHintMockTest::getTestCaseName);

TEST_P(SecPropAndDefaultPerfHintMockTest, SecPropAndDefaultPerfHintTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    Config config;
    bool bIsAuto = false;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos) {
        bIsAuto = true;
        plugin->SetName("AUTO");
    }

    if (device.find("MULTI") != std::string::npos)
        plugin->SetName("MULTI");

    for (auto& deviceName : targetDevices) {
        bool isCPUHelper = false;
        if (deviceName.find("CPU") != std::string::npos && bIsAuto) {
            auto item = config.find("MULTI_DEVICE_PRIORITIES");
            if (item != config.end() && item->second.find("GPU,CPU") != std::string::npos) {
                isCPUHelper = true;
            }
        }
        std::string HW_PerfHint;
        if (isCPUHelper) {
            // if it is CPU Helper, CPU should keep perf hint to LATENCY.
            HW_PerfHint = "LATENCY";
        } else {
            // HW default perf_hint
            HW_PerfHint = bIsAuto ? "LATENCY" : "THROUGHPUT";
        }
        auto item = config.find(deviceName);
        if (item != config.end() && !isCPUHelper) {
            // do not pass default perf_hint to HW
            HW_PerfHint = "No PERFORMANCE_HINT";
        }

        EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(deviceName),
                        ::testing::Matcher<const std::map<std::string, std::string>&>(ComparePerfHint(HW_PerfHint))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, config));
}

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiMock_SecPropAndDefaultPerfHintToHWTest,
                         SecPropAndDefaultPerfHintMockTest,
                         ::testing::ValuesIn(SecPropAndDefaultPerfHintMockTest::CreateSecPropAndDefaultPerfHintTestConfigs()),
                         SecPropAndDefaultPerfHintMockTest::getTestCaseName);
