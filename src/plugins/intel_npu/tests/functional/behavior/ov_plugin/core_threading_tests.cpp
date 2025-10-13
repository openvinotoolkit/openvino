// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core_threading_tests.hpp"
#include <utility>
#include "intel_npu/npu_private_properties.hpp"

namespace {

const Params params[] = {
    std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU,
                               {{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}}},
    std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU,
                               {{ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}}}};

const Params params_disable_umd_cache[] = {std::tuple<Device, Config>{
    ov::test::utils::DEVICE_NPU,
    {{ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT), ov::intel_npu::bypass_umd_caching(true)}}}};

const Params params_cached[] = {std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU, {}}};

}  // namespace

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_CoreThreadingTest_NPU,
                         CoreThreadingTestNPU,
                         testing::ValuesIn(params),
                         ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CoreThreadingTest_NPU,
                         CoreThreadingTestsWithIterNPU,
                         testing::Combine(testing::ValuesIn(params), testing::Values(15), testing::Values(50)),
                         ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestsWithIterNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CoreThreadingTest_NPU,
                         CoreThreadingTestsWithCacheEnabledNPU,
                         testing::Combine(testing::ValuesIn(params_cached), testing::Values(10), testing::Values(30)),
                         ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestsWithCacheEnabledNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_CoreThreadingTest_UmdCacheDisabled_NPU,
                         CoreThreadingTestsWithIterNPU,
                         testing::Combine(testing::ValuesIn(params_disable_umd_cache),
                                          testing::Values(8),
                                          testing::Values(20)),
                         ov::test::utils::appendPlatformTypeTestName<CoreThreadingTestsWithIterNPU>);
