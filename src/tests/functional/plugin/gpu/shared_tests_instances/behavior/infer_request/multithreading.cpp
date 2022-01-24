// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/multithreading.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> autoconfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU}},
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, std::string(CommonTestUtils::DEVICE_CPU) + "," + CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> auto_batch_configs = {
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                ::testing::Values(std::map<std::string, std::string>({}))),
                        InferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(configs)),
                        InferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(autoconfigs)),
                        InferRequestMultithreadingTests::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, InferRequestMultithreadingTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(auto_batch_configs)),
                         InferRequestMultithreadingTests::getTestCaseName);
}  // namespace
