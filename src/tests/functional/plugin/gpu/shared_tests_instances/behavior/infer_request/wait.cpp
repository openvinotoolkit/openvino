// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/wait.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU}}
    };

    const std::vector<std::map<std::string, std::string>> autoConfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
            {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
             CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU}}
    };


    const std::vector<std::map<std::string, std::string>> autoBatchConfigs = {
            // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , std::string(CommonTestUtils::DEVICE_GPU) + "(4)"},
                    // no timeout to avoid increasing the test time
                    {CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "0 "}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestWaitTests,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::Values(std::map<std::string, std::string>({}))),
                            InferRequestWaitTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestWaitTests,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(configs)),
                            InferRequestWaitTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestWaitTests,
                             ::testing::Combine(
                                     ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                     ::testing::ValuesIn(autoConfigs)),
                             InferRequestWaitTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, InferRequestWaitTests,
                             ::testing::Combine(
                                     ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                     ::testing::ValuesIn(autoBatchConfigs)),
                             InferRequestWaitTests::getTestCaseName);

}  // namespace
