// Copyright (C) 2018-2022 Intel Corporation
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
    INSTANTIATE_TEST_SUITE_P(
            ie_plugin,
            DefaultConfigurationTest,
            ::testing::Combine(
                    ::testing::ValuesIn(return_all_possible_device_combination(false)),
                    ::testing::Values(DefaultParameter{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(NO)})),
            DefaultConfigurationTest::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> pluginConfigs = {
            {{}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "10"}},
            // check that hints doesn't override customer value (now for streams and later for other config opts)
    };

    const std::vector<std::map<std::string, std::string>> pluginMultiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}}
    };

    INSTANTIATE_TEST_SUITE_P(ie_plugin, CorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(ov::test::conformance::targetDevice),
                ::testing::ValuesIn(pluginConfigs)),
            CorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ie_plugin_Hetero, CorrectConfigTests,
                         ::testing::Combine(
                                 ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                 ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_HETERO, pluginConfigs))),
                         CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Multi, CorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_MULTI, pluginMultiConfigs))),
            CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Auto, CorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_AUTO, pluginMultiConfigs))),
            CorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_AutoBatch, CorrectConfigTests,
                ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                    ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_BATCH, pluginConfigs))),
                CorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> inPluginConfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "NAN"}}
    };

    const std::vector<std::map<std::string, std::string>> pluginMultiInConfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}}
    };

    INSTANTIATE_TEST_SUITE_P(ie_plugin, IncorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(ov::test::conformance::targetDevice),
                ::testing::ValuesIn(inPluginConfigs)),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Hetero, IncorrectConfigTests,
             ::testing::Combine(
                     ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                     ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_HETERO, inPluginConfigs))),
             IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Multi, IncorrectConfigTests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_MULTI, pluginMultiInConfigs))),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Auto, IncorrectConfigTests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_AUTO, pluginMultiInConfigs))),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_AutoBatch, IncorrectConfigTests,
             ::testing::Combine(
                     ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                     ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_BATCH, pluginMultiInConfigs))),
             IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::Values(ov::test::conformance::targetDevice),
            ::testing::ValuesIn(inPluginConfigs)),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Hetero, IncorrectConfigAPITests,
             ::testing::Combine(
                     ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                     ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_HETERO, inPluginConfigs))),
             IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Multi, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_MULTI, pluginMultiInConfigs))),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_Auto, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_AUTO, pluginMultiInConfigs))),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(ie_plugin_AutoBatch, IncorrectConfigAPITests,
             ::testing::Combine(
                     ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                     ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_BATCH, inPluginConfigs))),
             IncorrectConfigAPITests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> pluginConfigsCheck = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "10"}}
    };

    INSTANTIATE_TEST_SUITE_P(ie_plugin, CorrectConfigCheck,
                             ::testing::Combine(
                                     ::testing::ValuesIn(return_all_possible_device_combination()),
                                     ::testing::ValuesIn(pluginConfigsCheck)),
                             CorrectConfigCheck::getTestCaseName);
} // namespace
