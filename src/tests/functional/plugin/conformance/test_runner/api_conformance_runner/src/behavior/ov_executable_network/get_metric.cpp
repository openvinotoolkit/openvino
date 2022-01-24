// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"
#include "openvino/runtime/core.hpp"
#include "api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;
using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//



INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassImportExportTestP,
        ::testing::Values(generateComplexDeviceName(CommonTestUtils::DEVICE_HETERO)));

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetConfigTest, OVClassExecutableNetworkGetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkSetConfigTest, OVClassExecutableNetworkSetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

////
//// Hetero Executable Network GetMetric
////

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values(ConformanceTests::targetDevice));

//////////////////////////////////////////////////////////////////////////////////////////

} // namespace

