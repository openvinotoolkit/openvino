// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <string>
#include <vector>

#include "behavior/plugin/core_integration.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassBasicTestP, IEClassBasicTestP,
        ::testing::Values(std::make_pair("openvino_template_plugin", CommonTestUtils::DEVICE_TEMPLATE)));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));


//
// IE Class SetConfig
//

class IEClassSetConfigTestHETERO : public BehaviorTestsUtils::IEClassNetworkTest,
                                   public BehaviorTestsUtils::IEPluginTestBase {
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        IEPluginTestBase::SetUp();
    }
};

TEST_F(IEClassSetConfigTestHETERO, smoke_SetConfigNoThrow) {
    {
        InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
        InferenceEngine::Parameter p;

        ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), CONFIG_VALUE(YES)}}, "HETERO"));
        ASSERT_NO_THROW(p = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)));
        bool dump = p.as<bool>();

        ASSERT_TRUE(dump);
    }

    {
        InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
        InferenceEngine::Parameter p;

        ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), CONFIG_VALUE(NO)}}, "HETERO"));
        ASSERT_NO_THROW(p = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)));
        bool dump = p.as<bool>();

        ASSERT_FALSE(dump);
    }

    {
        InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
        InferenceEngine::Parameter p;

        ASSERT_NO_THROW(ie.GetMetric("HETERO", METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
        ASSERT_NO_THROW(ie.SetConfig({{HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), CONFIG_VALUE(YES)}}, "HETERO"));
        ASSERT_NO_THROW(p = ie.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)));
        bool dump = p.as<bool>();

        ASSERT_TRUE(dump);
    }
}

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

class IEClassGetConfigTestTEMPLATE : public BehaviorTestsUtils::IEClassNetworkTest,
                                     public BehaviorTestsUtils::IEPluginTestBase {
    void SetUp() override {
        IEClassNetworkTest::SetUp();
        IEPluginTestBase::SetUp();
    }
};

TEST_F(IEClassGetConfigTestTEMPLATE, smoke_GetConfigNoThrow) {
    InferenceEngine::Core ie = BehaviorTestsUtils::createIECoreWithTemplate();
    InferenceEngine::Parameter p;
    std::string deviceName = CommonTestUtils::DEVICE_TEMPLATE;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)));
    std::vector<std::string> configValues = p;

    for (auto &&confKey : configValues) {
        if (CONFIG_KEY(DEVICE_ID) == confKey) {
            std::string defaultDeviceID = ie.GetConfig(deviceName, CONFIG_KEY(DEVICE_ID));
            std::cout << CONFIG_KEY(DEVICE_ID) << " : " << defaultDeviceID << std::endl;
        } else if (CONFIG_KEY(PERF_COUNT) == confKey) {
            auto defaultPerfCount = ie.GetConfig(deviceName, CONFIG_KEY(PERF_COUNT)).as<bool>();
            std::cout << CONFIG_KEY(PERF_COUNT) << " : " << defaultPerfCount << std::endl;
        } else if (CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) == confKey) {
            auto defaultExclusive = ie.GetConfig(deviceName, CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)).as<bool>();
            std::cout << CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS) << " : " << defaultExclusive << std::endl;
        }
    }
}

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));
} // namespace
