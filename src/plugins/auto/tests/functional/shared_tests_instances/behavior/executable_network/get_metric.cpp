// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"

using namespace BehaviorTestsDefinitions;

using namespace InferenceEngine::PluginConfigParams;

namespace {

//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values("AUTO:TEMPLATE", "MULTI:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values("AUTO:TEMPLATE", "MULTI:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
                         ::testing::Values("MULTI:TEMPLATE", "AUTO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::Values("MULTI:TEMPLATE", "AUTO:TEMPLATE"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassExecutableNetworkGetMetricTest,
                         IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
                         ::testing::Values("MULTI:TEMPLATE", "AUTO:TEMPLATE"));
}  // namespace
