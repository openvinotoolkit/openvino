// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
auto auto_batch_inconfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE}, {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "-1"}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE},
         {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, "DOESN'T EXIST"}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE},
         {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT, InferenceEngine::PluginConfigParams::LATENCY},
         {InferenceEngine::PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS, "-1"}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE},
         {InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "unknown_file"}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE},
         {InferenceEngine::PluginConfigParams::KEY_DEVICE_ID, "DEVICE_UNKNOWN"}}};
};

auto auto_batch_configs = []() {
    return std::vector<std::map<std::string, std::string>>{
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE}},
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), CommonTestUtils::DEVICE_TEMPLATE}, {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "1"}},
    };
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inconfigs())),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_inconfigs())),
                         IncorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         CorrectConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(auto_batch_configs())),
                         CorrectConfigTests::getTestCaseName);

}  // namespace