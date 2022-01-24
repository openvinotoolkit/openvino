// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
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
        smoke_OVClassCommon, OVClassBasicTestP,
        ::testing::Values(std::make_pair(getPluginLibNameByDevice(ConformanceTests::targetDevice), ConformanceTests::targetDevice)));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassNetworkTestP, OVClassNetworkTestP,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassImportExportTestP,
        ::testing::Values(generateComplexDeviceName(ConformanceTests::targetDevice)));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values(ConformanceTests::targetDevice));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_ThrowUnsupported,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest_ThrowUnsupported,
        ::testing::ValuesIn(returnAllPossibleDeviceCombination()));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetAvailableDevices, OVClassGetAvailableDevices,
        ::testing::Values(ConformanceTests::targetDevice));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest,
        ::testing::Values(ConformanceTests::targetDevice));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest,
        ::testing::Values(ConformanceTests::targetDevice));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest,
        ::testing::Values(ConformanceTests::targetDevice));
} // namespace

