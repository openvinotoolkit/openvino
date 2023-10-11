// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "behavior/infer_request/memory_states.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
std::vector<memoryStateParams> memoryStateTestCases = {
    memoryStateParams(InferRequestVariableStateTest::getNetwork(), {"c_1-3", "r_1-3"}, ov::test::utils::DEVICE_CPU, {}),
    memoryStateParams(InferRequestVariableStateTest::getNetwork(),
                      {"c_1-3", "r_1-3"},
                      ov::test::utils::DEVICE_HETERO,
                      {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_CPU}})};

INSTANTIATE_TEST_SUITE_P(smoke_VariableStateBasic,
                         InferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         InferRequestVariableStateTest::getTestCaseName);
} // namespace
