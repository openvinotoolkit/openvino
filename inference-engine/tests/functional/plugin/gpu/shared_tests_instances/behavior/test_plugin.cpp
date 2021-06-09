// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/test_plugin.hpp"
#include "gpu/gpu_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::I16,
            InferenceEngine::Precision::I32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_GPU}}
    };

    const std::vector<std::map<std::string, std::string>> AutoConfigs = {
            {{ AUTO_CONFIG_KEY(DEVICE_LIST) , CommonTestUtils::DEVICE_GPU}}
    };

    const std::vector<std::map<std::string, std::string>> auto_cpu_gpu_conf = {
            {{InferenceEngine::KEY_AUTO_DEVICE_LIST , std::string(CommonTestUtils::DEVICE_CPU) + "," + CommonTestUtils::DEVICE_GPU}}
    };

    const std::vector<std::map<std::string, std::string>> configsInput = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}}
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigsInputOutput = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}}
    };

    const std::vector<std::map<std::string, std::string>> configsOutput = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::GPU_THROUGHPUT_AUTO}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, BehaviorTestOutput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::ValuesIn(configsOutput)),
                            BehaviorTestOutput::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, BehaviorTestOutput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigsInputOutput)),
                            BehaviorTestOutput::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, BehaviorTestOutput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(AutoConfigs)),
                            BehaviorTestOutput::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_AutoCG_BehaviorTests, BehaviorTestOutput,
                            ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(auto_cpu_gpu_conf)),
                            BehaviorTestOutput::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, BehaviorTests,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::ValuesIn(configs)),
                            BehaviorTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, BehaviorTests,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigs)),
                            BehaviorTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, BehaviorTests,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(AutoConfigs)),
                            BehaviorTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_AutoCG_BehaviorTests, BehaviorTests,
                            ::testing::Combine(
                                ::testing::Values(InferenceEngine::Precision::FP32),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(auto_cpu_gpu_conf)),
                            BehaviorTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, BehaviorTestInput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                    ::testing::ValuesIn(configsInput)),
                            BehaviorTestInput::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, BehaviorTestInput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigsInputOutput)),
                            BehaviorTestInput::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, BehaviorTestInput,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(AutoConfigs)),
                            BehaviorTestInput::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_AutoCG_BehaviorTests, BehaviorTestInput,
                            ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(auto_cpu_gpu_conf)),
                            BehaviorTestInput::getTestCaseName);

}  // namespace
