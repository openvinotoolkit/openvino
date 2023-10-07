// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_plugin_config.hpp"
#include "ie_system_conf.h"
#include "behavior/plugin/configuration_tests.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
    #if (defined(__APPLE__) || defined(_WIN32))
    auto defaultBindThreadParameter = InferenceEngine::Parameter{[] {
        auto numaNodes = InferenceEngine::getAvailableNUMANodes();
        auto coreTypes = InferenceEngine::getAvailableCoresTypes();
        if (coreTypes.size() > 1) {
                return std::string{CONFIG_VALUE(HYBRID_AWARE)};
        } else if (numaNodes.size() > 1) {
            return std::string{CONFIG_VALUE(NUMA)};
        } else {
            return std::string{CONFIG_VALUE(NO)};
        }
    }()};
    #else
    auto defaultBindThreadParameter = InferenceEngine::Parameter{[] {
        auto coreTypes = InferenceEngine::getAvailableCoresTypes();
        if (coreTypes.size() > 1) {
                return std::string{CONFIG_VALUE(HYBRID_AWARE)};
        } else {
                return std::string{CONFIG_VALUE(YES)};
        }
    }()};
    #endif

    INSTANTIATE_TEST_SUITE_P(
            smoke_Basic,
            DefaultConfigurationTest,
            ::testing::Combine(
            ::testing::Values("CPU"),
            ::testing::Values(DefaultParameter{CONFIG_KEY(CPU_BIND_THREAD), defaultBindThreadParameter})),
            DefaultConfigurationTest::getTestCaseName);

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
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "8"}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::YES}},
            // check that hints doesn't override customer value (now for streams and later for other config opts)
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "3"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "3"}},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(Configs)),
            CorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> inconfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "OFF"}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, "OFF"}},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
            ::testing::Combine(
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(inconfigs)),
            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
            ::testing::Values(ov::test::utils::DEVICE_CPU),
            ::testing::ValuesIn(inconfigs)),
            IncorrectConfigAPITests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> ConfigsCheck = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "8"}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES}},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, CorrectConfigCheck,
                             ::testing::Combine(
                                     ::testing::Values(ov::test::utils::DEVICE_CPU),
                                     ::testing::ValuesIn(ConfigsCheck)),
                             CorrectConfigCheck::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> cpu_prop_config = {{
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
        {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES},
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "2"},
        {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO},
    }};

    const std::vector<std::map<std::string, std::string>> cpu_loadNetWork_config = {{
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
        {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::NO},
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "10"},
        {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES},
    }};

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                             SetPropLoadNetWorkGetPropTests,
                             ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                                ::testing::ValuesIn(cpu_prop_config),
                                                ::testing::ValuesIn(cpu_loadNetWork_config)),
                             SetPropLoadNetWorkGetPropTests::getTestCaseName);
} // namespace
