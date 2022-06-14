// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include <openvino/runtime/intel_auto/properties.hpp>

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {

const std::vector<ov::AnyMap> gpu_properties = {
        {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::ValuesIn(gpu_properties)),
        OVPropertiesDefaultTests::getTestCaseName);

const std::vector<ov::AnyMap> auto_multi_properties = {
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::intel_auto::device_bind_buffer("YES")},
        {ov::device::priorities(CommonTestUtils::DEVICE_GPU), ov::intel_auto::device_bind_buffer("NO")}
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiBehaviorTests, OVPropertiesTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO, CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(auto_multi_properties)),
        OVPropertiesTests::getTestCaseName);
} // namespace
