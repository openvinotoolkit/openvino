// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU}}
    };

    const std::vector<std::map<std::string, std::string>> autoConfigs = {
            {{InferenceEngine::KEY_AUTO_DEVICE_LIST, CommonTestUtils::DEVICE_GPU}}
    };

    const std::vector<std::map<std::string, std::string>> auto_cpu_gpu_conf = {
        {{InferenceEngine::KEY_AUTO_DEVICE_LIST , std::string(CommonTestUtils::DEVICE_CPU) + "," + CommonTestUtils::DEVICE_GPU}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::ValuesIn(configs)),
                            InferConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            InferConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(autoConfigs)),
                            InferConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_AutoCG_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(auto_cpu_gpu_conf)),
                            InferConfigTests::getTestCaseName);
}  // namespace
