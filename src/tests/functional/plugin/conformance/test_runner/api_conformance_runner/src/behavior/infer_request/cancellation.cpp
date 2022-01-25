// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/cancellation.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCancellationTests,
                         ::testing::Combine(
                                 ::testing::Values(targetDevice),
                                 ::testing::ValuesIn(std::vector<std::map<std::string, std::string>>{})),
                         InferRequestCancellationTests::getTestCaseName);
}  // namespace
