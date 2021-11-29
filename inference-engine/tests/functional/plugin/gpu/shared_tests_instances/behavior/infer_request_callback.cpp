// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_callback.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> autoConfigs = {
        {{ AUTO_CONFIG_KEY(DEVICE_LIST) , CommonTestUtils::DEVICE_GPU}}
};

const std::vector<std::map<std::string, std::string>> auto_cpu_gpu_conf = {
    {{InferenceEngine::KEY_AUTO_DEVICE_LIST , std::string(CommonTestUtils::DEVICE_CPU) + "," + CommonTestUtils::DEVICE_GPU}}
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CallbackTests,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GPU),
            ::testing::ValuesIn(configs)),
        CallbackTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CallbackTests,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(multiConfigs)),
        CallbackTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, CallbackTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                            ::testing::ValuesIn(autoConfigs)),
                        CallbackTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_AutoCG_BehaviorTests, CallbackTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                            ::testing::ValuesIn(auto_cpu_gpu_conf)),
                        CallbackTests::getTestCaseName);
}  // namespace
