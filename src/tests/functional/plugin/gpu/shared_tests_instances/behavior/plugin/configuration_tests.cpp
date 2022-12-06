// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"
#include "cldnn/cldnn_config.hpp"
#include "gpu/gpu_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    INSTANTIATE_TEST_SUITE_P(
            smoke_Basic,
            DefaultConfigurationTest,
            ::testing::Combine(
                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                    ::testing::Values(DefaultParameter{GPU_CONFIG_KEY(PLUGIN_THROTTLE), InferenceEngine::Parameter{std::string{"2"}}})),
            DefaultConfigurationTest::getTestCaseName);

    IE_SUPPRESS_DEPRECATED_START
    const std::vector<std::map<std::string, std::string>> inconfigs = {
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
            {{InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, "OFF"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}
    };

    const std::vector<std::map<std::string, std::string>> multiinconfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}}
    };


    const std::vector<std::map<std::string, std::string>> autoinconfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "NAN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "ABC"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                    {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "NAN"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "-1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
                {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "ABC"}}
    };


    const std::vector<std::map<std::string, std::string>> auto_batch_inconfigs = {
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_GPU},
                    {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "-1"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "ON"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DUMP_KERNELS, "ON"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_TUNING_MODE, "TUNING_UNKNOWN_MODE"}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU},
                    {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}
    };


    IE_SUPPRESS_DEPRECATED_END

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigTests,
                             ::testing::Combine(
                                     ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                     ::testing::ValuesIn(inconfigs)),
                             IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(multiinconfigs)),
                            IncorrectConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(autoinconfigs)),
                            IncorrectConfigTests::getTestCaseName);


    INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, IncorrectConfigTests,
             ::testing::Combine(
                     ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                     ::testing::ValuesIn(auto_batch_inconfigs)),
             IncorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> conf = {
            {}
    };

    IE_SUPPRESS_DEPRECATED_START
    const std::vector<std::map<std::string, std::string>> conf_gpu = {
            // Deprecated
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE, "0"}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_THROTTLE, "1"}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY, "0"}},
            {{InferenceEngine::CLDNNConfigParams::KEY_CLDNN_PLUGIN_PRIORITY, "1"}},

            {{InferenceEngine::GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE, "0"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE, "1"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY, "0"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY, "1"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_MAX_NUM_THREADS, "1"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_MAX_NUM_THREADS, "4"}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, InferenceEngine::PluginConfigParams::YES}},
            {{InferenceEngine::GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
                    {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            // check that hints doesn't override customer value (now for streams/throttling and later for other config opts)
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
             {InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, "3"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, "3"}},
            {{InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
             {InferenceEngine::GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE, "0"}}
    };
    IE_SUPPRESS_DEPRECATED_END

    const std::vector<std::map<std::string, std::string>> autoConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_GPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
             {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_NONE}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_ERROR}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_WARNING}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_INFO}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_DEBUG}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_TRACE}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES ,
                 CommonTestUtils::DEVICE_GPU + std::string(",") + CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW}}
    };

    const std::vector<std::map<std::string, std::string>> auto_batch_configs = {
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU}},
            {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , CommonTestUtils::DEVICE_GPU},
             {CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "1"}},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, DefaultValuesConfigTests,
            ::testing::Combine(
                ::testing::Values(CommonTestUtils::DEVICE_GPU),
                ::testing::ValuesIn(conf)),
            DefaultValuesConfigTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::Values(CommonTestUtils::DEVICE_GPU),
                    ::testing::ValuesIn(inconfigs)),
             IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                    ::testing::ValuesIn(multiinconfigs)),
            IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, IncorrectConfigAPITests,
            ::testing::Combine(
                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                    ::testing::ValuesIn(autoinconfigs)),
            IncorrectConfigAPITests::getTestCaseName);
    INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, IncorrectConfigAPITests,
             ::testing::Combine(
                     ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                     ::testing::ValuesIn(auto_batch_inconfigs)),
             IncorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests, CorrectConfigTests,
             ::testing::Combine(
                     ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                     ::testing::ValuesIn(auto_batch_configs)),
             CorrectConfigTests::getTestCaseName);

    const std::vector<std::map<std::string, std::string>> ExcluAsyncReqConfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU},
         {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES}},
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU},
         {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::NO}}};

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                             ExclusiveAsyncReqTests,
                             ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                                ::testing::ValuesIn(ExcluAsyncReqConfigs)),
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

    const std::vector<std::map<std::string, std::string>> auto_multi_prop_config = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU},
         {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
         {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES},
         {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "2"},
         {InferenceEngine::PluginConfigParams::KEY_ALLOW_AUTO_BATCHING, InferenceEngine::PluginConfigParams::NO},
         {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}}};

    const std::vector<std::map<std::string, std::string>> auto_multi_loadNetWork_config = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_GPU},
         {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
         {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::NO},
         {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "10"},
         {InferenceEngine::PluginConfigParams::KEY_ALLOW_AUTO_BATCHING, InferenceEngine::PluginConfigParams::YES},
         {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}}};

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                             SetPropLoadNetWorkGetPropTests,
                             ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_GPU),
                                                ::testing::ValuesIn(gpu_prop_config),
                                                ::testing::ValuesIn(gpu_loadNetWork_config)),
                             SetPropLoadNetWorkGetPropTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                             SetPropLoadNetWorkGetPropTests,
                             ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                                ::testing::ValuesIn(auto_multi_prop_config),
                                                ::testing::ValuesIn(auto_multi_loadNetWork_config)),
                             SetPropLoadNetWorkGetPropTests::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                             SetPropLoadNetWorkGetPropTests,
                             ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                                ::testing::ValuesIn(auto_multi_prop_config),
                                                ::testing::ValuesIn(auto_multi_loadNetWork_config)),
                             SetPropLoadNetWorkGetPropTests::getTestCaseName);
} // namespace
