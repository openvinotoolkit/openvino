// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/multithreading.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestMultithreadingTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                ::testing::Values(std::map<std::string, std::string>({}))),
        InferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestMultithreadingTests,
        ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(configs)),
        InferRequestMultithreadingTests::getTestCaseName);
}  // namespace
