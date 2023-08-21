// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"
#include "gpu/gpu_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    INSTANTIATE_TEST_SUITE_P(
            smoke_Basic,
            DefaultConfigurationTest,
            ::testing::Combine(
                    ::testing::Values(ov::test::utils::DEVICE_GPU),
                    ::testing::Values(DefaultParameter{GPU_CONFIG_KEY(PLUGIN_THROTTLE), InferenceEngine::Parameter{std::string{"2"}}})),
            DefaultConfigurationTest::getTestCaseName);

    IE_SUPPRESS_DEPRECATED_START
    auto inconfigs = []() {
        return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT,
              InferenceEngine::PluginConfigParams::THROUGHPUT},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, "OFF"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}};
    };

    auto multiinconfigs = []() {
        return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}}};
    };

    auto autoinconfigs = []() {
        return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "NAN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "ABC"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "NAN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
              ov::test::utils::DEVICE_GPU + std::string(",") + ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "ABC"}}};
    };

    auto auto_batch_inconfigs = []() {
        return std::vector<std::map<std::string, std::string>>{
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}};
    };

    IE_SUPPRESS_DEPRECATED_END

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
                             ::testing::Combine(
                                     ::testing::Values(ov::test::utils::DEVICE_GPU),
                                     ::testing::ValuesIn(inconfigs())),
                             IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                ::testing::ValuesIn(multiinconfigs())),
                            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                ::testing::ValuesIn(autoinconfigs())),
                            IncorrectConfigTests::getTestCaseName);


    INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, IncorrectConfigTests,
             ::testing::Combine(
                     ::testing::Values(ov::test::utils::DEVICE_BATCH),
                     ::testing::ValuesIn(auto_batch_inconfigs())),
             IncorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> conf = {
            {}
    };

    auto auto_batch_configs = []() {
        return std::vector<std::map<std::string, std::string>>{
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_GPU}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), ov::test::utils::DEVICE_GPU},
             {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1"}},
        };
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, DefaultValuesConfigTests,
            ::testing::Combine(
                ::testing::Values(ov::test::utils::DEVICE_GPU),
                ::testing::ValuesIn(conf)),
            DefaultValuesConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::Values(ov::test::utils::DEVICE_GPU),
                    ::testing::ValuesIn(inconfigs())),
             IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::Values(ov::test::utils::DEVICE_MULTI),
                    ::testing::ValuesIn(multiinconfigs())),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::Values(ov::test::utils::DEVICE_AUTO),
                    ::testing::ValuesIn(autoinconfigs())),
            IncorrectConfigAPITests::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, IncorrectConfigAPITests,
             ::testing::Combine(
                     ::testing::Values(ov::test::utils::DEVICE_BATCH),
                     ::testing::ValuesIn(auto_batch_inconfigs())),
             IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, CorrectConfigTests,
             ::testing::Combine(
                     ::testing::Values(ov::test::utils::DEVICE_BATCH),
                     ::testing::ValuesIn(auto_batch_configs())),
             CorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> gpu_prop_config = {{
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
        {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES},
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "2"},
        {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO},
    }};

    const std::vector<std::map<std::string, std::string>> gpu_loadNetWork_config = {{
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
        {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::NO},
        {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "10"},
        {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES},
    }};

    auto auto_multi_prop_config = []() {
        return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT,
              InferenceEngine::PluginConfigParams::THROUGHPUT},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
              InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED}}};
    };

    auto auto_multi_loadNetWork_config = []() {
        return std::vector<std::map<std::string, std::string>>{
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
              InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH}}};
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                             SetPropLoadNetWorkGetPropTests,
                             ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                                ::testing::ValuesIn(gpu_prop_config),
                                                ::testing::ValuesIn(gpu_loadNetWork_config)),
                             SetPropLoadNetWorkGetPropTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                             SetPropLoadNetWorkGetPropTests,
                             ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                                ::testing::ValuesIn(auto_multi_prop_config()),
                                                ::testing::ValuesIn(auto_multi_loadNetWork_config())),
                             SetPropLoadNetWorkGetPropTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                             SetPropLoadNetWorkGetPropTests,
                             ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                                ::testing::ValuesIn(auto_multi_prop_config()),
                                                ::testing::ValuesIn(auto_multi_loadNetWork_config())),
                             SetPropLoadNetWorkGetPropTests::getTestCaseName);
} // namespace
