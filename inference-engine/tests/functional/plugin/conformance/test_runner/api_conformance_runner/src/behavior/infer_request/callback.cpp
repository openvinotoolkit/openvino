// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/callback.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace BehaviorTestsDefinitions;
using namespace ConformanceTests;

const std::vector<std::map<std::string, std::string>> configsCallback = {
        {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(targetDevice),
                                 ::testing::ValuesIn(configsCallback)),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                 ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                 ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestCallbackTests::getTestCaseName);
}  // namespace
