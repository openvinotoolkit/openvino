// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;
using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassBasicTestP,
        ::testing::Values(std::make_pair(get_plugin_lib_name_by_device(ov::test::conformance::targetDevice), ov::test::conformance::targetDevice)));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassNetworkTestP,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassImportExportTestP,
        ::testing::Values(generate_complex_device_name(ov::test::conformance::targetDevice)));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetAvailableDevices,
        ::testing::ValuesIn(return_all_possible_device_combination()));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassGetConfigTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassQueryNetworkTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        ov_plugin, OVClassLoadNetworkTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));
} // namespace

