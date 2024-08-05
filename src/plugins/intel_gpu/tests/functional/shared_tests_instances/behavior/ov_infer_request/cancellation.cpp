// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/cancellation.hpp"

using namespace ov::test::behavior;

namespace {
auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestCancellationTests,
        ::testing::Combine(
            ::testing::Values(ov::test::utils::DEVICE_GPU),
            ::testing::ValuesIn(configs())),
        OVInferRequestCancellationTests::getTestCaseName);
}  // namespace
