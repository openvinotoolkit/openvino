// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace BehaviorTestsDefinitions;

INSTANTIATE_TEST_SUITE_P(ie_infer_request, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::ValuesIn(empty_config)),
                         InferRequestPerfCountersTest::getTestCaseName);

}  // namespace
