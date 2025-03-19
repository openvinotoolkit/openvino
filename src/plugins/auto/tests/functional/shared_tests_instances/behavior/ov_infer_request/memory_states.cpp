// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/memory_states.hpp"

using namespace ov::test::behavior;

namespace {
std::vector<memoryStateParams> memoryStateTestCases = {
    memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                      {"c_1-3", "r_1-3"},
                      ov::test::utils::DEVICE_AUTO,
                      {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}}),
    memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                      {"c_1-3", "r_1-3"},
                      ov::test::utils::DEVICE_MULTI,
                      {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}})};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         OVInferRequestVariableStateTest::getTestCaseName);
}  // namespace