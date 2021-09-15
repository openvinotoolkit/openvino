// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/ov_core_integration.hpp"
#include "openvino/runtime/core.hpp"

using namespace BehaviorTestsDefinitions;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCommon, OVClassBasicTestP,
        ::testing::Values(std::make_pair("clDNNPlugin", "GPU")));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassNetworkTestP, OVClassNetworkTestP,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassImportExportTestP,
        ::testing::Values("HETERO:GPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetAvailableDevices, OVClassGetAvailableDevices,
        ::testing::Values("GPU"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest,
        ::testing::Values("GPU"));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest,
        ::testing::Values("GPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest,
        ::testing::Values("GPU"));
} // namespace

