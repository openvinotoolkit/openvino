// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/multithreading.hpp"
#include "ie_plugin_config.hpp"

#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace ConformanceTests;
using namespace BehaviorTestsDefinitions;

const std::vector<std::map<std::string, std::string>> configsMultithreading = {
        {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(configsMultithreading)),
                         InferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestMultithreadingTests::getTestCaseName);

}  // namespace
