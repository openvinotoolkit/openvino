// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
auto configs = []() {
    return std::vector<std::map<std::string, std::string>>{{}};
};

auto AutoBatchConfigs =
    []() {
        return std::vector<std::map<std::string, std::string>>{
            // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_GPU) + "(4)"},
             // no timeout to avoid increasing the test time
             {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
    };

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(configs())),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(AutoBatchConfigs())),
                         InferRequestPerfCountersTest::getTestCaseName);
}  // namespace
