// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace ConformanceTests;
using namespace BehaviorTestsDefinitions;

const std::vector<std::map<std::string, std::string>> configsPerfCounters = {
        {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(configsPerfCounters)),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestPerfCountersTest::getTestCaseName);


}  // namespace
