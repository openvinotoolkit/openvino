// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/test_plugin.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_MYRIAD}}
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigsInputOutput = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_MYRIAD}}
    };

    const std::vector<InferenceEngine::Precision> netPrecision = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, BehaviorTestOutput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecision),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            BehaviorTestOutput::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, BehaviorTestOutput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecision),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigsInputOutput)),
                            BehaviorTestOutput::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, BehaviorTests,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            BehaviorTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, BehaviorTests,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigs)),
                            BehaviorTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, BehaviorTestInput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            BehaviorTestInput::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, BehaviorTestInput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigsInputOutput)),
                            BehaviorTestInput::getTestCaseName);

}  // namespace
