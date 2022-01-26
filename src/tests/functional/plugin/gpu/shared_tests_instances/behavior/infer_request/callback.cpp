// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/callback.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> autoConfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
        {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU}}
};

const std::vector<std::map<std::string, std::string>> autoBatchConfigs = {
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCallbackTests,
        ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_GPU),
            ::testing::ValuesIn(configs)),
        InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestCallbackTests,
        ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(multiConfigs)),
        InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestCallbackTests,
        ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
            ::testing::ValuesIn(autoConfigs)),
        InferRequestCallbackTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, InferRequestCallbackTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                 ::testing::ValuesIn(autoBatchConfigs)),
                         InferRequestCallbackTests::getTestCaseName);
}  // namespace
