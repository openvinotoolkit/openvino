// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_plugin_config.hpp"
#include "ie_system_conf.h"
#include "behavior/plugin/configuration_tests.hpp"
#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

namespace {
    #if (defined(__APPLE__) || defined(_WIN32))
    auto defaultBindThreadParameter = InferenceEngine::Parameter{[] {
        auto numaNodes = InferenceEngine::getAvailableNUMANodes();
        if (numaNodes.size() > 1) {
            return std::string{CONFIG_VALUE(NUMA)};
        } else {
            return std::string{CONFIG_VALUE(NO)};
        }
    }()};
    #else
    auto defaultBindThreadParameter = InferenceEngine::Parameter{std::string{CONFIG_VALUE(YES)}};
    #endif

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> conf = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> Configs = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "10"}},
            // check that hints doesn't override customer value (now for streams and later for other config opts)
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(ConformanceTests::targetDevice),
                ::testing::ValuesIn(Configs)),
            CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI, MultiConfigs))),
            CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, CorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO, MultiConfigs))),
            CorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> inconfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "NAN"}}
    };

    const std::vector<std::map<std::string, std::string>> multiinconfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "NAN"}}
    };

    const std::vector<std::map<std::string, std::string>> multiconf = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(ConformanceTests::targetDevice),
                ::testing::ValuesIn(inconfigs)),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI, multiinconfigs))),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO, multiinconfigs))),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::Values(ConformanceTests::targetDevice),
            ::testing::ValuesIn(inconfigs)),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI, multiinconfigs))),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
            ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO, multiinconfigs))),
            IncorrectConfigAPITests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> ConfigsCheck = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "10"}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigCheck,
                             ::testing::Combine(
                                     ::testing::Values(ConformanceTests::targetDevice),
                                     ::testing::ValuesIn(ConfigsCheck)),
                             CorrectConfigCheck::getTestCaseName);
} // namespace
