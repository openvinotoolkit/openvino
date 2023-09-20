// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/core_integration.hpp"

using namespace BehaviorTestsDefinitions;

using namespace InferenceEngine::PluginConfigParams;

// defined in plugin_name.cpp
extern const char* cpu_plugin_file_name;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//
//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_IEClassGetMetricTest,
                         IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassGetMetricTest,
                         IEClassGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassGetMetricTest,
                         IEClassGetMetricTest_FULL_DEVICE_NAME,
                         ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassGetMetricTest,
                         IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
                         ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassGetMetricTest,
                         IEClassGetMetricTest_ThrowUnsupported,
                         ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(smoke_IEClassGetConfigTest,
                         IEClassGetConfigTest_ThrowUnsupported,
                         ::testing::Values("MULTI", "AUTO"));
//////////////////////////////////////////////////////////////////////////////////////////
}  // namespace
