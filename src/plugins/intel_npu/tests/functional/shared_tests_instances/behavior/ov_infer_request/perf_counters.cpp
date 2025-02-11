// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                               ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
                                              {ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                               ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

const std::vector<ov::AnyMap> autoConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                              ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
                                             {ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                              ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Multi_BehaviorTests,
                         OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Auto_BehaviorTests,
                         OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

}  // namespace
