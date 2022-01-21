// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"
#include "api_conformance_helpers.hpp"

using namespace ov::test::conformance;
using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine::PluginConfigParams;

namespace {

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassImportExportTestP, IEClassImportExportTestP,
        ::testing::Values(generateComplexDeviceName(CommonTestUtils::DEVICE_HETERO)));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values(ConformanceTests::targetDevice));

} // namespace