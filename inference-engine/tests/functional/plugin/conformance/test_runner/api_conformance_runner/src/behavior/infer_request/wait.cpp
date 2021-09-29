// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/wait.hpp"
#include "ie_plugin_config.hpp"
#include "conformance.hpp"

namespace {

using namespace ConformanceTests;
using namespace BehaviorTestsDefinitions;

const std::vector<std::map<std::string, std::string>> configsWait = {
        {},
};

const std::vector<std::map<std::string, std::string>> MulticonfigsWait = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , targetDevice}}
};

const std::vector<std::map<std::string, std::string>> AutoconfigsWait = {
        {{ AUTO_CONFIG_KEY(DEVICE_LIST) , targetDevice}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(configsWait)),
                         InferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(MulticonfigsWait)),
                         InferRequestWaitTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoconfigsWait)),
                         InferRequestWaitTests::getTestCaseName);

}  // namespace
