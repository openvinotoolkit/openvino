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
#include "include/mock_auto_device_plugin.hpp"
#include "include/mock_common.hpp"

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::StrEq;
using Config = std::map<std::string, std::string>;
using ConfigParams = std::tuple<std::string, Config>;
using namespace MockMultiDevice;

namespace {
void custom_unsetenv(const char *name) {
#ifdef _WIN32
    _putenv((std::string(name) + "=").c_str());
#else
    ::unsetenv(name);
#endif
}
}

class AutoSetLogLevel : public ::testing::TestWithParam<ConfigParams> {
public:
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::CNNNetwork cnnNet;
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;

    // mock exeNetwork helper
    ov::SoPtr<IExecutableNetworkInternal> mockExeNetwork;
    std::vector<DeviceInformation> metaDevices;
    std::shared_ptr<NiceMock<MockIInferRequestInternal>> inferReqInternal;
    size_t optimalNum;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string log_level;
        Config config;
        std::tie(log_level, config) = obj.param;
        std::ostringstream result;
        result << log_level;
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
        mockExeNetwork = {};
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
        ON_CALL(*core, GetMetric(_, StrEq(METRIC_KEY(SUPPORTED_CONFIG_KEYS)), _)).WillByDefault(Return(supportConfigs));
        ON_CALL(*core, GetConfig(_, StrEq(ov::compilation_num_threads.name()))).WillByDefault(Return(12));
        ON_CALL(*core,
                LoadNetwork(::testing::Matcher<const InferenceEngine::CNNNetwork&>(_),
                            ::testing::Matcher<const std::string&>(_),
                            ::testing::Matcher<const Config&>(_)))
            .WillByDefault(Return(mockExeNetwork));

        metaDevices = {{CommonTestUtils::DEVICE_CPU, {}, -1}, {CommonTestUtils::DEVICE_GPU, {}, -1}};
        // DeviceInformation devInfo;
        ON_CALL(*plugin, ParseMetaDevices(_, _)).WillByDefault(Return(metaDevices));
        ON_CALL(*plugin, GetValidDevice)
            .WillByDefault([](const std::vector<DeviceInformation>& metaDevices, const std::string& netPrecision) {
                std::list<DeviceInformation> devices(metaDevices.begin(), metaDevices.end());
                return devices;
            });
        ON_CALL(*plugin, SelectDevice(_, _, _)).WillByDefault(Return(metaDevices[1]));
    }
};

TEST_P(AutoSetLogLevel, setLogLevelFromConfig) {
    custom_unsetenv("OPENVINO_LOG_LEVEL");
    std::string log_level;
    Config config;
    std::tie(log_level, config) = this->GetParam();
    plugin->SetName("AUTO");
    plugin->LoadExeNetworkImpl(cnnNet, config);
    int a = 0;
    DEBUG_RUN([&a](){a++;});
    INFO_RUN([&a](){a++;});
    if (log_level == "LOG_DEBUG" || log_level == "LOG_TRACE") {
        EXPECT_EQ(a, 2);
    } else if (log_level == "LOG_INFO") {
        EXPECT_EQ(a, 1);
    } else {
        EXPECT_EQ(a, 0);
    }
}

const std::vector<ConfigParams> testConfigs = {ConfigParams{"LOG_NONE", {{"LOG_LEVEL", "LOG_NONE"}}},
                                               ConfigParams{"LOG_ERROR", {{"LOG_LEVEL", "LOG_ERROR"}}},
                                               ConfigParams{"LOG_WARNING", {{"LOG_LEVEL", "LOG_WARNING"}}},
                                               ConfigParams{"LOG_INFO", {{"LOG_LEVEL", "LOG_INFO"}}},
                                               ConfigParams{"LOG_DEBUG", {{"LOG_LEVEL", "LOG_DEBUG"}}},
                                               ConfigParams{"LOG_TRACE", {{"LOG_LEVEL", "LOG_TRACE"}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         AutoSetLogLevel,
                         ::testing::ValuesIn(testConfigs),
                         AutoSetLogLevel::getTestCaseName);
