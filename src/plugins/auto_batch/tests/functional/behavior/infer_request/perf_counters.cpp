// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
auto AutoBatchConfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
        // explicit batch size 4 to avoid fallback to no auto-batching
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
         // no timeout to avoid increasing the test time
         {ov::auto_batch_timeout.name(), "0"}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         InferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(AutoBatchConfigs())),
                         InferRequestPerfCountersTest::getTestCaseName);

}  // namespace