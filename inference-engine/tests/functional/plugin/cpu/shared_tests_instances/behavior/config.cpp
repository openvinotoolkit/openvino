// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_plugin_config.hpp"
#include "behavior/config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> conf = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> Configs = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "8"}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "10"}}
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "8"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "10"}}
    };

    const std::vector<std::map<std::string, std::string>> AutoConfigs = {
            {{InferenceEngine::KEY_AUTO_DEVICE_LIST , CommonTestUtils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigTests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(Configs)),
            CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(MultiConfigs)),
            CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, CorrectConfigTests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
            ::testing::ValuesIn(AutoConfigs)),
            CorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> inconfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "OFF"}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, "OFF"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "NAN"}}
    };

    const std::vector<std::map<std::string, std::string>> multiinconfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "OFF"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, "OFF"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "NAN"}}
    };

    const std::vector<std::map<std::string, std::string>> autoinconfigs = {
        {{InferenceEngine::KEY_AUTO_DEVICE_LIST , CommonTestUtils::DEVICE_CPU},
            {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "OFF"}}
    };

    const std::vector<std::map<std::string, std::string>> multiconf = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigAPITests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(conf)),
            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigAPITests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(multiconf)),
            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, CorrectConfigAPITests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
            ::testing::ValuesIn(AutoConfigs)),
            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(inconfigs)),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(multiinconfigs)),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::ValuesIn(inconfigs)),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(multiinconfigs)),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, IncorrectConfigAPITests,
                            ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(autoinconfigs)),
                            IncorrectConfigAPITests::getTestCaseName);

} // namespace