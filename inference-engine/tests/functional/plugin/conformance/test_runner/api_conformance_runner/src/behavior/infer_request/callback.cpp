// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/callback.hpp"
#include "conformance.hpp"

namespace {
using namespace BehaviorTestsDefinitions;
using namespace ConformanceTests;

const std::vector<std::map<std::string, std::string>> configsCallback = {
        {},
};

const std::vector<std::map<std::string, std::string>> generateMultiConfigsCallback() {
        return {{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), targetDevice}}};
}

const std::vector<std::map<std::string, std::string>> generateAutoConfigsCallback() {
        return {{{AUTO_CONFIG_KEY(DEVICE_LIST), targetDevice}}};
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(targetDevice),
                                 ::testing::ValuesIn(configsCallback)),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                 ::testing::ValuesIn(generateMultiConfigsCallback())),
                         InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                 ::testing::ValuesIn(generateAutoConfigsCallback())),
                         InferRequestCallbackTests::getTestCaseName);
}  // namespace
