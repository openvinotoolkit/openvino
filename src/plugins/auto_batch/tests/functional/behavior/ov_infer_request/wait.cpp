// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/wait.hpp"

using namespace ov::test::behavior;

namespace {
auto AutoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{
        // explicit batch size 4 to avoid fallback to no auto-batching
        {{ov::device::priorities.name(), std::string(ov::test::utils::DEVICE_TEMPLATE) + "(4)"},
         // no timeout to avoid increasing the test time
         {ov::auto_batch_timeout(0)}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         OVInferRequestWaitTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(AutoBatchConfigs())),
                         OVInferRequestWaitTests::getTestCaseName);

}  // namespace