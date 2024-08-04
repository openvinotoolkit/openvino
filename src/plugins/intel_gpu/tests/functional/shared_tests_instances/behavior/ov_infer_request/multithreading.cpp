// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/multithreading.hpp"

using namespace ov::test::behavior;

namespace {

auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
        {ov::num_streams(ov::streams::AUTO)},
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                            OVInferRequestMultithreadingTests::getTestCaseName);
}  // namespace
