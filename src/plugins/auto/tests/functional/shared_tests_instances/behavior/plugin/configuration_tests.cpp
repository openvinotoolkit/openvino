// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"

#include "ie_plugin_config.hpp"
#include "ie_system_conf.h"

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

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> conf = {{}};

const std::vector<std::map<std::string, std::string>> MultiConfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::NO}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}}};

const std::vector<std::map<std::string, std::string>> AutoConfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_NONE}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_ERROR}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_WARNING}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_INFO}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_DEBUG}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, InferenceEngine::PluginConfigParams::LOG_TRACE}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
      InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
      InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
      InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW}}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         CorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(MultiConfigs)),
                         CorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         CorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(AutoConfigs)),
                         CorrectConfigTests::getTestCaseName);

const std::vector<std::map<std::string, std::string>> multiinconfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, "OFF"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, "OFF"}},
};

const std::vector<std::map<std::string, std::string>> autoinconfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "should be int"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "OFF"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, "OFF"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "-1"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY, "ABC"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_LOG_LEVEL, "NAN"}}};

const std::vector<std::map<std::string, std::string>> multiconf = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "1"}},
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE}}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiinconfigs)),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoinconfigs)),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiinconfigs)),
                         IncorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoinconfigs)),
                         IncorrectConfigAPITests::getTestCaseName);

const std::vector<std::map<std::string, std::string>> auto_multi_prop_config = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
     {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
      InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED}}};

const std::vector<std::map<std::string, std::string>> auto_multi_loadNetWork_config = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE},
     {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::THROUGHPUT},
     {InferenceEngine::PluginConfigParams::KEY_MODEL_PRIORITY,
      InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH}}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         SetPropLoadNetWorkGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_prop_config),
                                            ::testing::ValuesIn(auto_multi_loadNetWork_config)),
                         SetPropLoadNetWorkGetPropTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SetPropLoadNetWorkGetPropTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(auto_multi_prop_config),
                                            ::testing::ValuesIn(auto_multi_loadNetWork_config)),
                         SetPropLoadNetWorkGetPropTests::getTestCaseName);
}  // namespace
