// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
                         ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
                         ::testing::Values("GPU", "MULTI:GPU", "HETERO:GPU", "AUTO:GPU,CPU", "BATCH:GPU"));

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetConfigTest,
                         OVClassExecutableNetworkGetConfigTest,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkSetConfigTest,
                         OVClassExecutableNetworkSetConfigTest,
                         ::testing::Values("GPU"));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassHeteroExecutableNetworlGetMetricTest,
                         OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassHeteroExecutableNetworlGetMetricTest,
                         OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassHeteroExecutableNetworlGetMetricTest,
                         OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
                         ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassHeteroExecutableNetworlGetMetricTest,
                         OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
                         ::testing::Values("GPU"));

} // namespace

