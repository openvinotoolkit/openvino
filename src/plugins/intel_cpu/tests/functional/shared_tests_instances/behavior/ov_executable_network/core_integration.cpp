// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"
#include "behavior/ov_executable_network/hetero_properties.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassCompiledModelProperties_SupportedProperties,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassCompiledModelGetIncorrectProperties,
        ::testing::Values("CPU", "MULTI:CPU", "HETERO:CPU", "AUTO:CPU"));

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>> GetMetricTest_ExecutionDevice_CPU = {
        {"CPU", std::make_pair(ov::AnyMap{}, "CPU")}};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_EXEC_DEVICES,
        ::testing::ValuesIn(GetMetricTest_ExecutionDevice_CPU),
        OVCompileModelGetExecutionDeviceTests::getTestCaseName);

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVCompiledModelGetSupportedPropertiesTest, OVCompiledModelGetSupportedPropertiesTest,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCompiledModelUnsupportedConfigTest, OVClassCompiledModelUnsupportedConfigTest,
        ::testing::Combine(::testing::Values("CPU"),
                           ::testing::Values(std::make_pair("unsupported_config", "some_value"))));

////
//// Hetero Executable Network GetMetric
////
//
//INSTANTIATE_TEST_SUITE_P(
//        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
//        ::testing::Values("CPU"));
//
//INSTANTIATE_TEST_SUITE_P(
//        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
//        ::testing::Values("CPU"));
//
//INSTANTIATE_TEST_SUITE_P(
//        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
//        ::testing::Values("CPU"));
//
//INSTANTIATE_TEST_SUITE_P(
//        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
//        ::testing::Values("CPU"));
INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_EXEC_DEVICES,
        ::testing::Values("CPU"));
//////////////////////////////////////////////////////////////////////////////////////////

} // namespace

