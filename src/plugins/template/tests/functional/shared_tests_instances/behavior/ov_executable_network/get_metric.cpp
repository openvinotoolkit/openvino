// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;

using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//



INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassExecutableNetworkImportExportTestP,
        ::testing::Values("HETERO:TEMPLATE"));

//
// Executable Network GetMetric
//

std::vector<std::string> devices = {"TEMPLATE", "MULTI:TEMPLATE", "HETERO:TEMPLATE", "AUTO:TEMPLATE"};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetMetricTest, OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::ValuesIn(devices));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:TEMPLATE"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));        

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_TEMPLATE)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)),
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY::getTestCaseName);

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkGetConfigTest, OVClassExecutableNetworkGetConfigTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkSetConfigTest, OVClassExecutableNetworkSetConfigTest,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

////
//// Hetero Executable Network GetMetric
////
//
INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
       ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
       ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
       ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(
       smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
       ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));
INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassHeteroExecutableNetworkGetMetricTest, OVClassHeteroExecutableNetworkGetMetricTest_EXEC_DEVICES,
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE));
//////////////////////////////////////////////////////////////////////////////////////////

} // namespace

