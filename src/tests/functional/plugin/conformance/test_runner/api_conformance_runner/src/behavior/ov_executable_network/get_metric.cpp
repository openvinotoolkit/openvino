// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_api_conformance_helpers.hpp"


namespace {
using namespace ov::test::behavior;
using namespace ov::test::conformance;
using namespace InferenceEngine::PluginConfigParams;
//
// IE Class Common tests with <pluginName, deviceName params>
//



INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassImportExportTestP,
        ::testing::ValuesIn(return_all_possible_device_combination()));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination()));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkGetConfigTest,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassExecutableNetworkSetConfigTest,
        ::testing::Values(ov::test::conformance::targetDevice));

////
//// Hetero Executable Network GetMetric
////

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::ValuesIn(return_all_possible_device_combination()));


} // namespace

