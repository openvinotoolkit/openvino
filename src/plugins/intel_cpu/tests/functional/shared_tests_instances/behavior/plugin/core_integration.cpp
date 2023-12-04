// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/core_integration.hpp"

using namespace BehaviorTestsDefinitions;

using namespace InferenceEngine::PluginConfigParams;

// defined in plugin_name.cpp
extern const char * cpu_plugin_file_name;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassCommon, IEClassBasicTestP,
        ::testing::Values(std::make_pair(cpu_plugin_file_name, "CPU")));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values("CPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values("CPU"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values("CPU"));

//////////////////////////////////////////////////////////////////////////////////////////

TEST(IEClassBasicTest, smoke_SetConfigAfterCreatedThrow) {
    InferenceEngine::Core ie;
    std::string value = {};

    ASSERT_NO_THROW(ie.SetConfig({{KEY_CPU_THREADS_NUM, "1"}}, "CPU"));
    ASSERT_NO_THROW(value = ie.GetConfig("CPU", KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("1", value);

    ASSERT_NO_THROW(ie.SetConfig({{KEY_CPU_THREADS_NUM, "4"}}, "CPU"));
    ASSERT_NO_THROW(value = ie.GetConfig("CPU", KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("4", value);
}

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values("CPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTestWithThrow,
        ::testing::Values(""));
} // namespace
