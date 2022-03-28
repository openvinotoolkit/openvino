// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/core_integration.hpp"
#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine::PluginConfigParams;
using namespace ov::test::conformance;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassCommon, IEClassBasicTestP,
        ::testing::Values(std::make_pair(get_plugin_lib_name_by_device(ov::test::conformance::targetDevice), ov::test::conformance::targetDevice)));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values(ov::test::conformance::targetDevice));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values(ov::test::conformance::targetDevice));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values(ov::test::conformance::targetDevice));

//////////////////////////////////////////////////////////////////////////////////////////

TEST(IEClassBasicTest, smoke_SetConfigAfterCreatedThrow) {
    InferenceEngine::Core ie;
    std::string value = {};

    ASSERT_NO_THROW(ie.SetConfig({{KEY_CPU_THREADS_NUM, "1"}}, ov::test::conformance::targetDevice));
    ASSERT_NO_THROW(value = ie.GetConfig(ov::test::conformance::targetDevice, KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("1", value);

    ASSERT_NO_THROW(ie.SetConfig({{KEY_CPU_THREADS_NUM, "4"}}, ov::test::conformance::targetDevice));
    ASSERT_NO_THROW(value = ie.GetConfig(ov::test::conformance::targetDevice, KEY_CPU_THREADS_NUM).as<std::string>());
    ASSERT_EQ("4", value);
}

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values(ov::test::conformance::targetDevice));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values(ov::test::conformance::targetDevice));
} // namespace