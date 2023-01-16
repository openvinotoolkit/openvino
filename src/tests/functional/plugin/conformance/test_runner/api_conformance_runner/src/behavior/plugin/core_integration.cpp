// Copyright (C) 2018-2023 Intel Corporation
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
        ie_plugin, IEClassBasicTestP,
        ::testing::ValuesIn(generate_pairs_plugin_name_by_device()));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassNetworkTestP,
        ::testing::ValuesIn(return_all_possible_device_combination()));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetAvailableDevices,
        ::testing::Values(targetDevice));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassGetConfigTest,
        ::testing::ValuesIn(return_all_possible_device_combination(false)));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassQueryNetworkTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        ie_plugin, IEClassLoadNetworkTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));
} // namespace
