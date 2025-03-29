// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/memory_states.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_infer_request_mandatory,
                         OVInferRequestVariableStateTest,
                         ::testing::Values(memoryStateParams(OVInferRequestVariableStateTest::get_network(),
                                                             {"c_1-3", "r_1-3"},
                                                             ov::test::utils::target_device,
                                                             ov::AnyMap({}))),
                         OVInferRequestVariableStateTest::getTestCaseName);
}  // namespace