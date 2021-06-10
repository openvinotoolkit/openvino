// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/config.hpp"
#include <template/template_config.hpp>

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
    {{TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
    {{TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS), "8"}},
};

const std::vector<std::map<std::string, std::string>> inconfigs = {
    {{TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS), CONFIG_VALUE(NO)}},
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(inconfigs)),
                        IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(inconfigs)),
                        IncorrectConfigAPITests::getTestCaseName);


INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigAPITests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        CorrectConfigAPITests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                ::testing::ValuesIn(configs)),
                        CorrectConfigAPITests::getTestCaseName);

} // namespace