// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/wait.hpp"

using namespace ov::test::behavior;

namespace {

auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
        {ov::num_streams(ov::streams::AUTO)},
    };
};

auto AutoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{// explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
                                   {ov::device::priorities(std::string(ov::test::utils::DEVICE_GPU) + "(4)"),
                                    // no timeout to avoid increasing the test time
                                    ov::auto_batch_timeout(0)}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                            OVInferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, OVInferRequestWaitTests,
                         ::testing::Combine(
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(AutoBatchConfigs())),
                            OVInferRequestWaitTests::getTestCaseName);
}  // namespace
