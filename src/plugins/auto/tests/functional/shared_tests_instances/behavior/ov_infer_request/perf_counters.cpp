// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> Multiconfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU), ov::enable_profiling(true)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
         ov::intel_auto::device_bind_buffer(false)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
         ov::intel_auto::device_bind_buffer(true)}
};

const std::vector<ov::AnyMap> Autoconfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU), ov::enable_profiling(true)},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
                                    ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                                    ov::intel_auto::device_bind_buffer(true)}
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                ::testing::ValuesIn(Multiconfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                ::testing::ValuesIn(Autoconfigs)),
                         OVInferRequestPerfCountersTest::getTestCaseName);
}  // namespace
