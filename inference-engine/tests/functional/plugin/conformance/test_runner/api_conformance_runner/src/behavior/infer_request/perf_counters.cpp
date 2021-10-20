// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/perf_counters.hpp"
#include "conformance.hpp"

namespace {

using namespace ConformanceTests;
using namespace BehaviorTestsDefinitions;

const std::vector<std::map<std::string, std::string>> configsPerfCounters = {
        {}
};

const std::vector<std::map<std::string, std::string>> generateMulticonfigsPerfCounters() {
        return {{{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), targetDevice }}};
}

const std::vector<std::map<std::string, std::string>> generateAutoconfigsPerfCounters() {
        return {{{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), targetDevice }}};
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(configsPerfCounters)),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(generateMulticonfigsPerfCounters())),
                         InferRequestPerfCountersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPerfCountersTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(generateAutoconfigsPerfCounters())),
                         InferRequestPerfCountersTest::getTestCaseName);


}  // namespace
