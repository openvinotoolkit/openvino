// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/infer_request/memory_states.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/builders.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
std::vector<memoryStateParams> memoryStateTestCases = {
    memoryStateParams(InferRequestVariableStateTest::getNetwork(), {"c_1-3", "r_1-3"}, CommonTestUtils::DEVICE_CPU, {}),
    memoryStateParams(InferRequestVariableStateTest::getNetwork(),
                      {"c_1-3", "r_1-3"},
                      CommonTestUtils::DEVICE_HETERO,
                      {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_CPU}})};

std::vector<memoryStateParams> memoryStateAutoTestCases = {
    memoryStateParams(InferRequestVariableStateTest::getNetwork(),
                      {"c_1-3", "r_1-3"},
                      CommonTestUtils::DEVICE_AUTO,
                      {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_CPU}})};

std::vector<memoryStateParams> memoryStateMultiTestCases = {
    memoryStateParams(InferRequestVariableStateTest::getNetwork(),
                      {"c_1-3", "r_1-3"},
                      CommonTestUtils::DEVICE_MULTI,
                      {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_CPU}})};

INSTANTIATE_TEST_SUITE_P(smoke_VariableStateBasic,
                         InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         InferRequestVariableStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateAutoTestCases),
                         InferRequestVariableStateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateMultiTestCases),
                         InferRequestVariableStateTest::getTestCaseName);
} // namespace
