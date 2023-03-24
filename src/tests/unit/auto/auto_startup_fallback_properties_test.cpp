// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include <ie_core.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ngraph_functions/subgraph_builders.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "plugin/mock_auto_device_plugin.hpp"
#include "mock_common.hpp"
#include <thread>

using ::testing::_;
using ::testing::StrEq;
using ::testing::Return;
using ::testing::NiceMock;
using Config = std::map<std::string, std::string>;
using namespace MockMultiDevice;

using ConfigParams = std::tuple<bool,
                                Config>;

// define a matcher if all the elements of subMap are contained in the map.
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
class AutoStartupFallback : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<ngraph::Function>               function;
    InferenceEngine::CNNNetwork                     cnnNet;
    std::shared_ptr<NiceMock<MockICore>>                      core;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;

    //mock exeNetwork helper
    ov::SoPtr<IExecutableNetworkInternal>  mockExeNetwork;
    // config for Auto device
    std::map<std::string, std::string>              config;
    std::vector<DeviceInformation>                  metaDevices;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>>     inferReqInternal;
    size_t optimalNum;

public:
    void TearDown() override {
        core.reset();
        plugin.reset();
        //mockIExeNet.reset();
        mockExeNetwork = {};
        config.clear();
        metaDevices.clear();
        inferReqInternal.reset();
    }

    void SetUp() override {
       // prepare mockExeNetwork
       auto mockIExeNet = std::make_shared<NiceMock<MockIExecutableNetworkInternal>>();
       mockExeNetwork = {mockIExeNet, {}};
       // prepare mockicore and cnnNetwork for loading
       core = std::make_shared<NiceMock<MockICore>>();
       NiceMock<MockMultiDeviceInferencePlugin>* mock_multi = new NiceMock<MockMultiDeviceInferencePlugin>();
       plugin.reset(mock_multi);
       function = ngraph::builder::subgraph::makeConvPoolRelu();
       cnnNet = InferenceEngine::CNNNetwork(function);
       // replace core with mock Icore
       plugin->SetCore(core);
       // mock execNetwork can work
       inferReqInternal = std::make_shared<NiceMock<MockIInferRequestInternal>>();
       ON_CALL(*mockIExeNet.get(), CreateInferRequest()).WillByDefault(Return(inferReqInternal));
       IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, optimalNum, 1);
       ON_CALL(*mockIExeNet.get(), GetMetric(StrEq(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS))))
           .WillByDefault(Return(optimalNum));
       IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, supportConfigs, {});
       ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _))
           .WillByDefault(Return(supportConfigs));
       ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name())))
           .WillByDefault(Return(12));
    }
};

TEST_P(AutoStartupFallback, releaseResource) {
    // get Parameter
    bool startup_fallback;
    Config config;
    std::tie(startup_fallback, config) = this->GetParam();
    // test auto plugin
    plugin->SetName("AUTO");

    ON_CALL(*core, LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                ::testing::Matcher<const std::string&>(_),
                ::testing::Matcher<const Config&>(_))).WillByDefault(Return(mockExeNetwork));

    metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, -1}, {CommonTestUtils::DEVICE_GPU, {}, -1}};
    // DeviceInformation devInfo;
    ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));
    ON_CALL(*plugin, GetValidDevice)
        .WillByDefault([this](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
            std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
            return devices;
        });
    ON_CALL(*plugin, SelectDevice(_, _, _)).WillByDefault(Return(metaDevices[1]));

    EXPECT_CALL(
        *core,
        LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                    ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_GPU),
                    ::testing::Matcher<const std::map<std::string, std::string>&>(_)))
        .Times(1);
    if (startup_fallback) {
        std::map<std::string, std::string> test_map = {{"PERFORMANCE_HINT", "LATENCY"}};
        EXPECT_CALL(
            *core,
            LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                        ::testing::Matcher<const std::string&>(CommonTestUtils::DEVICE_CPU),
                        ::testing::Matcher<const std::map<std::string, std::string>&>(MapContains(test_map))))
            .Times(1);
    }

    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(cnnNet, config));
}

const std::vector<ConfigParams> testConfigs = {ConfigParams {true, {{"ENABLE_STARTUP_FALLBACK", "YES"}}},
                                               ConfigParams {false, {{"ENABLE_STARTUP_FALLBACK", "NO"}}}
                                              };

INSTANTIATE_TEST_SUITE_P(smoke_Auto_StartupFallback,
                         AutoStartupFallback,
                         ::testing::ValuesIn(testConfigs));
