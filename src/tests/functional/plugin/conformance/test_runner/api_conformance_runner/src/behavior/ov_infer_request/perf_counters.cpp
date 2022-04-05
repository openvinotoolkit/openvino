// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_infer_request, OVInferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::ValuesIn(empty_ov_config)),
                         OVInferRequestPerfCountersTest::getTestCaseName);

}  // namespace
