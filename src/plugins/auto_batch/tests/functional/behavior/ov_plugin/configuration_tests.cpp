// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
auto auto_batch_inconfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
        {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE},
         {ov::hint::performance_mode.name(), "DOESN'T EXIST"}},
        {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE},
         {ov::hint::performance_mode.name(), "LATENCY"},
         {ov::hint::num_requests.name(), "-1"}},
        {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}, {"CONFIG_FILE", "unknown_file"}},
        {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}, {ov::device::id.name(), "DEVICE_UNKNOWN"}}};
};

auto auto_batch_configs = []() {
    return std::vector<std::map<std::string, std::string>>{
        {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}},
        {{ov::device::priorities.name(), ov::test::utils::DEVICE_TEMPLATE}, {ov::auto_batch_timeout.name(), "1"}},
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inconfigs())),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inconfigs())),
                         IncorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         CorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_configs())),
                         CorrectConfigTests::getTestCaseName);

}  // namespace