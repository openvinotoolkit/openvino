// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/memory_states.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
std::vector<memoryStateParams> memoryStateTestCases = {memoryStateParams(InferRequestVariableStateTest::getNetwork(),
                                                                         {"c_1-3", "r_1-3"},
                                                                         ov::test::utils::DEVICE_TEMPLATE,
                                                                         {})};

INSTANTIATE_TEST_SUITE_P(smoke_Template_BehaviorTests,
                         InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         InferRequestVariableStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Template_BehaviorTests,
                         InferRequestQueryStateExceptionTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         InferRequestQueryStateExceptionTest::getTestCaseName);
}  // namespace
