// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"

using namespace ov::test::behavior;

namespace {
auto configs = []() {
    return std::vector<ov::AnyMap>{{}};
};

auto Multiconfigs = []() {
    return std::vector<ov::AnyMap>{
        {ov::device::priorities(ov::test::utils::DEVICE_GPU)},
#ifdef ENABLE_INTEL_CPU
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU), ov::enable_profiling(true)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
         ov::intel_auto::device_bind_buffer(false)},
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
         ov::intel_auto::device_bind_buffer(true)}
#endif
    };
};

auto Autoconfigs = []() {
    return std::vector<ov::AnyMap>{{ov::device::priorities(ov::test::utils::DEVICE_GPU)},
#ifdef ENABLE_INTEL_CPU
                                   {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
                                    ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                                    ov::intel_auto::device_bind_buffer(true)}
#endif
    };
};

auto AutoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_GPU) + "(4)"},
         // no timeout to avoid increasing the test time
         {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                ::testing::ValuesIn(Multiconfigs())),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                ::testing::ValuesIn(Autoconfigs())),
                         OVInferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs())),
                         OVInferRequestPerfCountersTest::getTestCaseName);

auto MulticonfigsTest = []() {
    return std::vector<ov::AnyMap>{
#ifdef ENABLE_INTEL_CPU
        {ov::device::priorities(ov::test::utils::DEVICE_GPU, ov::test::utils::DEVICE_CPU),
         ov::device::priorities(ov::test::utils::DEVICE_CPU, ov::test::utils::DEVICE_GPU)}
#endif
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(MulticonfigsTest())),
                         OVInferRequestPerfCountersExceptionTest::getTestCaseName);
}  // namespace
