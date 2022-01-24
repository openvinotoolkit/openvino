// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/core_integration.hpp"
#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;
using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassCommon, IEClassBasicTestP,
        ::testing::Values(std::make_pair(getPluginLibNameByDevice(ConformanceTests::targetDevice), ConformanceTests::targetDevice)));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values(ConformanceTests::targetDevice));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values(ConformanceTests::targetDevice));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

//////////////////////////////////////////////////////////////////////////////////////////

TEST(IEClassBasicTest, smoke_SetConfigAfterCreatedThrow) {
    InferenceEngine::Core ie;
    std::string value = {};

    ASSERT_NO_THROW(ie.SetConfig({{KEY_CPU_THREADS_NUM, "1"}}, ConformanceTests::targetDevice));
    ASSERT_NO_THROW(value = ie.GetConfig(ConformanceTests::targetDevice, KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("1", value);

    ASSERT_NO_THROW(ie.SetConfig({{KEY_CPU_THREADS_NUM, "4"}}, ConformanceTests::targetDevice));
    ASSERT_NO_THROW(value = ie.GetConfig(ConformanceTests::targetDevice, KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("4", value);
}

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values(ConformanceTests::targetDevice));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values(ConformanceTests::targetDevice));
} // namespace