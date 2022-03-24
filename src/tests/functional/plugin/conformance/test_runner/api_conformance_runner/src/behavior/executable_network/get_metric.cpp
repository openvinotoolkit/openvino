// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"
#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine::PluginConfigParams;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassImportExportTestP, IEClassImportExportTestP,
        ::testing::Values(generate_complex_device_name(CommonTestUtils::DEVICE_HETERO)));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(return_all_possible_device_combination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(return_all_possible_device_combination()));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values(ov::test::conformance::targetDevice));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values(ov::test::conformance::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values(ov::test::conformance::targetDevice));

} // namespace