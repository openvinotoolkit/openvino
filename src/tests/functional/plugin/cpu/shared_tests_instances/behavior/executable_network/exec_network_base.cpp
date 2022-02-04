// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };
    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{ InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU}}
    };

    const std::vector<std::map<std::string, std::string>> heteroConfigs = {
            {{"TARGET_FALLBACK", CommonTestUtils::DEVICE_CPU}}};

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(multiConfigs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, ExecutableNetworkBaseTest,
                             ::testing::Combine(
                                     ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                     ::testing::ValuesIn(heteroConfigs)),
                             ExecutableNetworkBaseTest::getTestCaseName);

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::I16,
            InferenceEngine::Precision::U16
    };

    const std::vector<std::map<std::string, std::string>> configSetPrc = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };

    const std::vector<std::map<std::string, std::string>> AutoConfigsSetPrc = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU}},
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigsSetPrc = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configSetPrc)),
                            ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(MultiConfigsSetPrc)),
                            ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(AutoConfigsSetPrc)),
                            ExecNetSetPrecision::getTestCaseName);
}  // namespace
