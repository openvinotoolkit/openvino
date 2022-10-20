// Copyright (C) 2018-2022 Intel Corporation
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

#include "cpp/ie_plugin.hpp"
#include "mock_common.hpp"
#include "plugin/mock_auto_device_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"

using ::testing::_;
using ::testing::AllOf;
using ::testing::AtLeast;
using ::testing::Contains;
using ::testing::Eq;
using ::testing::InvokeWithoutArgs;
using ::testing::IsTrue;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::Pair;
using ::testing::Property;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::Throw;
using Config = std::map<std::string, std::string>;
MATCHER_P(MapContains, subMap, "Check if all the elements of the subMap are contained in the map.") {
    if (subMap.empty())
        return true;
    for (auto& item : subMap) {
        auto key = item.first;
        auto value = item.second;
        auto dest = arg.find(key);
        if (dest == arg.end()) {
            return false;
        } else if (dest->second != value) {
            return false;
        }
    }
    return true;
}
using namespace MockMultiDevice;

using ConfigParams = std::tuple<std::string,               // virtual device name to load network
                                std::vector<std::string>,  // hardware device name to expect loading network on
                                Config>;                   // secondary property setting to device

static std::vector<ConfigParams> testConfigs;

class LoadNetworkWithSecondaryConfigsMockTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<MockICore> core;
    std::shared_ptr<MockMultiDeviceInferencePlugin> plugin;
    InferenceEngine::CNNNetwork simpleCnnNetwork;
    // mock cpu exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> cpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal> cpuMockExeNetwork;
    MockIInferencePlugin* cpuMockIPlugin;
    InferenceEngine::InferencePlugin cpuMockPlugin;

    // mock actual exeNetwork
    std::shared_ptr<MockIExecutableNetworkInternal> gpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal> gpuMockExeNetwork;
    MockIInferencePlugin* gpuMockIPlugin;
    InferenceEngine::InferencePlugin gpuMockPlugin;
    std::shared_ptr<MockIInferRequestInternal> inferReqInternal;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string deviceName;
        std::vector<std::string> targetDevices;
        Config deviceConfigs;
        std::tie(deviceName, targetDevices, deviceConfigs) = obj.param;
        std::ostringstream result;
        if (deviceName.find("AUTO") != std::string::npos)
            result << "_target_mock_device_"
                   << "AUTO";
        else
            result << "_target_device_"
                   << "MULTI";
        for (auto& item : deviceConfigs) {
            result << "_device_" << item.first;
            // result << "_config_" << item.second;
        }
        return result.str();
    }

    static std::vector<ConfigParams> CreateConfigs() {
        testConfigs.clear();
        testConfigs.push_back(
            ConfigParams{"AUTO", {"CPU"}, {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO", {"CPU", "GPU"}, {{"GPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:CPU", {"CPU"}, {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:CPU,GPU", {"CPU"}, {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"AUTO:GPU", {"GPU"}, {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(ConfigParams{"AUTO:GPU,CPU",
                                           {"CPU", "GPU"},
                                           {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});

        testConfigs.push_back(
            ConfigParams{"MULTI:CPU", {"CPU"}, {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU"}}});
        testConfigs.push_back(ConfigParams{"MULTI:CPU,GPU",
                                           {"CPU", "GPU"},
                                           {{"CPU", "NUM_STREAMS 3"}, {"MULTI_DEVICE_PRIORITIES", "CPU,GPU"}}});
        testConfigs.push_back(
            ConfigParams{"MULTI:GPU", {"GPU"}, {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU"}}});
        testConfigs.push_back(ConfigParams{"MULTI:GPU,CPU",
                                           {"CPU", "GPU"},
                                           {{"GPU", "NUM_STREAMS 5"}, {"MULTI_DEVICE_PRIORITIES", "GPU,CPU"}}});
        return testConfigs;
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
        // prepare cpuMockExeNetwork
        cpuMockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
        auto cpuMockIPluginPtr = std::make_shared<MockIInferencePlugin>();
        ON_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _))
            .WillByDefault(Return(cpuMockIExeNet));
        cpuMockPlugin = InferenceEngine::InferencePlugin{cpuMockIPluginPtr, {}};
        // remove annoying ON CALL message
        EXPECT_CALL(*cpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        cpuMockExeNetwork = cpuMockPlugin.LoadNetwork(CNNNetwork{}, {});

        // prepare actualMockExeNetwork
        gpuMockIExeNet = std::make_shared<MockIExecutableNetworkInternal>();
        auto gpuMockIPluginPtr = std::make_shared<MockIInferencePlugin>();
        ON_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _))
            .WillByDefault(Return(gpuMockIExeNet));
        gpuMockPlugin = InferenceEngine::InferencePlugin{gpuMockIPluginPtr, {}};
        // remove annoying ON CALL message
        EXPECT_CALL(*gpuMockIPluginPtr, LoadNetwork(MatcherCast<const CNNNetwork&>(_), _)).Times(1);
        gpuMockExeNetwork = gpuMockPlugin.LoadNetwork(CNNNetwork{}, {});

        // prepare mockicore and cnnNetwork for loading
        core = std::shared_ptr<MockICore>(new MockICore());
        auto* origin_plugin = new MockMultiDeviceInferencePlugin();
        plugin = std::shared_ptr<MockMultiDeviceInferencePlugin>(origin_plugin);
        // replace core with mock Icore
        plugin->SetCore(core);
        // mock execNetwork can work
        inferReqInternal = std::make_shared<MockIInferRequestInternal>();
        ON_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
        ON_CALL(*gpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));

        ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));
        ON_CALL(*gpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));

        std::vector<std::string> availableDevs = {"CPU", "GPU"};
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));

        // IE_SET_METRIC(SUPPORTED_METRICS, metrics, {METRIC_KEY(SUPPORTED_CONFIG_KEYS)});
        std::vector<std::string> metrics = {METRIC_KEY(SUPPORTED_CONFIG_KEYS)};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_METRICS)), _)).WillByDefault(Return(metrics));

        std::vector<std::string> configKeys = {"SUPPORTED_CONFIG_KEYS", "NUM_STREAMS"};
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(configKeys));

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
        // IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, cpuCability, {"FP32", "FP16", "INT8", "BIN"});
        // IE_SET_METRIC(OPTIMIZATION_CAPABILITIES, gpuCability, {"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"});
        std::vector<std::string> cpuCability{"FP32", "FP16", "INT8", "BIN"};
        std::vector<std::string> gpuCability{"FP32", "FP16", "BATCHED_BLOB", "BIN", "INT8"};
        ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_CPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(RETURN_MOCK_VALUE(cpuCability));
        ON_CALL(*core, GetMetric(StrEq(CommonTestUtils::DEVICE_GPU), StrEq(METRIC_KEY(OPTIMIZATION_CAPABILITIES)), _))
            .WillByDefault(RETURN_MOCK_VALUE(gpuCability));

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

TEST_P(LoadNetworkWithSecondaryConfigsMockTest, LoadNetworkWithSecondaryConfigsTest) {
    std::string device;
    std::vector<std::string> targetDevices;
    Config config;
    std::tie(device, targetDevices, config) = this->GetParam();
    if (device.find("AUTO") != std::string::npos)
        plugin->SetName("AUTO");
    if (device.find("MULTI") != std::string::npos)
        plugin->SetName("MULTI");
    auto setExpectCall = [&](const std::string& deviceName) {
        std::stringstream strConfigs(config[deviceName]);
        Config deviceConfigs;
        ov::util::Read<Config>{}(strConfigs, deviceConfigs);
        // EXPECT_CALL(*core,
        //            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
        //                        ::testing::Matcher<const std::string&>(StrEq(deviceName)),
        //                        ::testing::Matcher<const std::map<std::string, std::string>&>(
        //                            MapContains(deviceConfigs))))
        //    .Times(1);
    };

    for (auto& deviceName : targetDevices) {
        auto item = config.find(deviceName);
        Config deviceConfigs;
        if (item != config.end()) {
            std::stringstream strConfigs(item->second);
            ov::util::Read<Config>{}(strConfigs, deviceConfigs);
        }
        EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(deviceName),
                        ::testing::Matcher<const std::map<std::string, std::string>&>(MapContains(deviceConfigs))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, config));
}

INSTANTIATE_TEST_SUITE_P(smoke_AutoMock_LoadNetworkWithSecondaryConfigs,
                         LoadNetworkWithSecondaryConfigsMockTest,
                         ::testing::ValuesIn(LoadNetworkWithSecondaryConfigsMockTest::CreateConfigs()));
// LoadNetworkWithSecondaryConfigsMockTest::getTestCaseName);
