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

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_GPU)}};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(nightly_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));


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

INSTANTIATE_TEST_SUITE_P(nightly_OVClassHeteroExecutableNetworlGetMetricTest,
                         OVClassHeteroExecutableNetworkGetMetricTest_EXEC_DEVICES,
                         ::testing::Values("GPU.0"));

const std::vector<DevicePropertiesNumStreamsParams> devicePropertiesConfigsNoThrow = {
    DevicePropertiesNumStreamsParams{"AUTO:GPU", {ov::device::properties("GPU", ov::num_streams(5))}, "GPU"},
    DevicePropertiesNumStreamsParams{"AUTO", {ov::device::properties("GPU", ov::num_streams(5))}, "GPU"},
    DevicePropertiesNumStreamsParams{"AUTO", {ov::device::properties("CPU", ov::num_streams(5))}, "CPU"},
    DevicePropertiesNumStreamsParams{"AUTO:GPU,CPU", {ov::device::properties("CPU", ov::num_streams(2))}, "CPU"},
    DevicePropertiesNumStreamsParams{"AUTO:GPU,CPU", {ov::device::properties("GPU", ov::num_streams(2))}, "GPU"}};

const std::vector<DevicePropertiesNumStreamsParams> devicePropertiesConfigsThrow = {
    DevicePropertiesNumStreamsParams{"AUTO:GPU", {ov::device::properties("CPU", ov::num_streams(2))}, "CPU"}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_OVClassAutoExcutableNetowrkGetDevicePropertiesTestNoThrow,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PROPERTIES,
                         ::testing::ValuesIn(devicePropertiesConfigsNoThrow),
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PROPERTIES::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_OVClassAutoExcutableNetowrkGetDevicePropertiesTestThrow,
                         OVClassExecutableNetworkGetMetricTestThrow_DEVICE_PROPERTIES,
                         ::testing::ValuesIn(devicePropertiesConfigsThrow),
                         OVClassExecutableNetworkGetMetricTestThrow_DEVICE_PROPERTIES::getTestCaseName);
} // namespace

