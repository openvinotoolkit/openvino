// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/memory_states.hpp"

using namespace ov::test::behavior;
using namespace ov;

namespace {
std::vector<memoryStateParams> memoryStateTestCases = {memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                                                                         {"c_1-3", "r_1-3"},
                                                                         ov::test::utils::DEVICE_CPU,
                                                                         {})};

INSTANTIATE_TEST_SUITE_P(smoke_VariableState,
                         OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         OVInferRequestVariableStateTest::getTestCaseName);
}  // namespace