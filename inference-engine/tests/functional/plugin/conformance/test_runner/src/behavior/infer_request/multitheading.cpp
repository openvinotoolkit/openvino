// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/multithreading.hpp"
#include "ie_plugin_config.hpp"

#include "conformance.hpp"

namespace {

using namespace ConformanceTests;
using namespace BehaviorTestsDefinitions;

const std::vector<std::map<std::string, std::string>> configsMultithreading = {
        {},
};

const std::vector<std::map<std::string, std::string>> MulticonfigsMultithreading = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), targetDevice }}
};

const std::vector<std::map<std::string, std::string>> AutoconfigsMultithreading = {
        {{ AUTO_CONFIG_KEY(DEVICE_LIST), targetDevice}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(configsMultithreading)),
                         InferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(MulticonfigsMultithreading)),
                         InferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoconfigsMultithreading)),
                         InferRequestMultithreadingTests::getTestCaseName);

}  // namespace
