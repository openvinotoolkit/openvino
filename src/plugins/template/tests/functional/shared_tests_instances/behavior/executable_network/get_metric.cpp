// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"

#include <string>
#include <utility>
#include <vector>

using namespace BehaviorTestsDefinitions;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE, "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE, "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE, "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE, "MULTI:TEMPLATE", "HETERO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
                         IEClassExecutableNetworkGetMetricTest,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE, "MULTI:TEMPLATE", "HETERO:TEMPLATE"));
//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetConfigTest,
                         IEClassExecutableNetworkGetConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkSetConfigTest,
                         IEClassExecutableNetworkSetConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_IEClassHeteroExecutableNetworlGetMetricTest,
                         IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassHeteroExecutableNetworlGetMetricTest,
                         IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassHeteroExecutableNetworlGetMetricTest,
                         IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassHeteroExecutableNetworlGetMetricTest,
                         IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));
}  // namespace
