// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/multithreading.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         InferRequestMultithreadingTests::getTestCaseName);
}  // namespace
