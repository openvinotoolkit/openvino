// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/memory_states.hpp"

namespace ov {
namespace test {
namespace behavior {
std::vector<memoryStateParams> memoryStateTestCases = {
    memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                      {"c_1-3", "r_1-3"},
                      ov::test::utils::DEVICE_HETERO,
                      {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)})};

INSTANTIATE_TEST_SUITE_P(smoke_VariableState,
                         OVInferRequestVariableStateTest,
                         ::testing::ValuesIn(memoryStateTestCases),
                         OVInferRequestVariableStateTest::getTestCaseName);
}  // namespace behavior
}  // namespace test
}  // namespace ov