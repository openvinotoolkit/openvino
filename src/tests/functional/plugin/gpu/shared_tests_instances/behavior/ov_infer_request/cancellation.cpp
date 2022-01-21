// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/cancellation.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCancellationTests,
        ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_GPU),
            ::testing::ValuesIn(configs)),
        OVInferRequestCancellationTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatchBehaviorTests, OVInferRequestCancellationTests,
                         ::testing::Combine(
                                 ::testing::Values(std::string(CommonTestUtils::DEVICE_BATCH) + ":" + CommonTestUtils::DEVICE_GPU),
                                 ::testing::ValuesIn(configs)),
                         OVInferRequestCancellationTests::getTestCaseName);
}  // namespace
