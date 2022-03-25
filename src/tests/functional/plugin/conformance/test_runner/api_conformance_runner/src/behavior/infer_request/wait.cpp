// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/wait.hpp"
#include "ie_plugin_config.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace BehaviorTestsDefinitions;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(std::vector<std::map<std::string, std::string>>{})),
                         InferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestWaitTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                 ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_HETERO))),
                         InferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Batch_BehaviorTests, InferRequestWaitTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_BATCH))),
                         InferRequestWaitTests::getTestCaseName);
}  // namespace
