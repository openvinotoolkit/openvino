// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/cancellation.hpp"
#include "conformance.hpp"

namespace {
using namespace BehaviorTestsDefinitions;
using namespace ConformanceTests;

const std::vector<std::map<std::string, std::string>> configsCancel = {
        {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCancellationTests,
                         ::testing::Combine(
                                 ::testing::Values(targetDevice),
                                 ::testing::ValuesIn(configsCancel)),
                         InferRequestCancellationTests::getTestCaseName);
}  // namespace
