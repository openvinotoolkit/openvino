// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common_test_utils/test_constants.hpp>
#include <ngraph_functions/subgraph_builders.hpp>

#include "plugin/mock_load_network_properties.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"

using ::testing::_;
using ::testing::MatcherCast;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrEq;
using ::testing::Throw;

using namespace MockMultiDevice;
using Config = std::map<std::string, std::string>;
using ConfigParams = std::tuple<bool, std::vector<std::string>>;

// define a matcher to check if perf hint expects
MATCHER_P(ComparePerfHint, perfHint, "Check if perf hint expects.") {
    std::string arg_perfHint = "";
    auto itor = arg.find(PluginConfigParams::KEY_PERFORMANCE_HINT);
    if (itor != arg.end()) {
        arg_perfHint = itor->second;
    }

    return perfHint == arg_perfHint;
}

class AutoCTPUTCallMulti : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>> plugin;
    InferenceEngine::CNNNetwork simpleCnnNetwork;
    // mock cpu exeNetwork
    std::shared_ptr<NiceMock<MockIExecutableNetworkInternal>> cpuMockIExeNet;
    ov::SoPtr<IExecutableNetworkInternal> cpuMockExeNetwork;
    NiceMock<MockIInferencePlugin>* cpuMockIPlugin;
    std::shared_ptr<InferenceEngine::IInferencePlugin> cpuMockPlugin;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>> inferReqInternal;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::vector<std::string> targetDevices;
        bool AutoCallMulti;
        std::tie(AutoCallMulti, targetDevices) = obj.param;
        std::ostringstream result;
        if (AutoCallMulti) {
            result << "AutoCallMulti_";
        } else {
            result << "Multi_";
        }
        result << "ctput_loadnetwork_to_device_";
        for (auto& device : targetDevices) {
            if (device == targetDevices.back()) {
                result << device;
            } else {
                result << device << "_";
            }
        }
        return result.str();
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

        // prepare mockicore and cnnNetwork for loading
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        auto* origin_plugin = new NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>();
        plugin = std::shared_ptr<NiceMock<MockMultiPluginForLoadNetworkWithPropertiesTest>>(origin_plugin);
        // replace core with mock Icore
        plugin->SetCore(core);
        inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
        ON_CALL(*cpuMockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
        ON_CALL(*cpuMockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
            .WillByDefault(Return("0"));

        std::vector<std::string> availableDevs = {"CPU", "GPU"};
        ON_CALL(*core, GetAvailableDevices()).WillByDefault(Return(availableDevs));

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
            .WillByDefault(Throw(InferenceEngine::GeneralError{""}));
        std::shared_ptr<ngraph::Function> simpleNetwork = ngraph::builder::subgraph::makeSingleConv();
        ASSERT_NO_THROW(simpleCnnNetwork = InferenceEngine::CNNNetwork(simpleNetwork));
    }
};

TEST_P(AutoCTPUTCallMulti, CTPUTDevicesLogicTest) {
    std::vector<std::string> targetDevices;
    std::string targetDev;
    bool AutoCallMulti;
    Config config;
    std::tie(AutoCallMulti, targetDevices) = this->GetParam();
    plugin->SetName("MULTI");
    for (auto& deviceName : targetDevices) {
        targetDev += deviceName;
        targetDev += ((deviceName == targetDevices.back()) ? "" : ",");
    }
    if (AutoCallMulti) {
        std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> exeNetwork;
        config.insert({{CONFIG_KEY(PERFORMANCE_HINT), InferenceEngine::PluginConfigParams::CUMULATIVE_THROUGHPUT}});
        config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, targetDev});
        EXPECT_CALL(*core,
                    LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_CPU),
                                ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .Times(1);
        EXPECT_CALL(*core,
                    LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_GPU),
                                ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .Times(1);
        ASSERT_NO_THROW(exeNetwork = plugin->LoadExeNetworkImpl(simpleCnnNetwork, config));
        EXPECT_EQ(exeNetwork->GetMetric(ov::execution_devices.name()).as<std::string>(), CommonTestUtils::DEVICE_CPU);
    } else {
        config.insert({InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, targetDev});
        EXPECT_CALL(*core,
                    LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_CPU),
                                ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .Times(0);
        EXPECT_CALL(*core,
                    LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                                ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_GPU),
                                ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
            .Times(1);
        EXPECT_THROW(plugin->LoadExeNetworkImpl(simpleCnnNetwork, config), IE::Exception);
    }
}

const std::vector<ConfigParams> testConfigs = {
    ConfigParams{false, {"CPU", "GPU"}},
    ConfigParams{false, {"GPU", "CPU"}},
    ConfigParams{true, {"CPU", "GPU"}},
    ConfigParams{true, {"GPU", "CPU"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoCTPUTCallMulti,
                         AutoCTPUTCallMulti,
                         ::testing::ValuesIn(testConfigs),
                         AutoCTPUTCallMulti::getTestCaseName);
