// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/configuration_tests.hpp"

#include <template/config.hpp>

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {{ov::template_plugin::throughput_streams.name(), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
    {{ov::template_plugin::throughput_streams.name(), InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
    {{ov::template_plugin::throughput_streams.name(), "8"}},
};

const std::vector<std::map<std::string, std::string>> inconfigs = {
    {{ov::template_plugin::throughput_streams.name(), CONFIG_VALUE(NO)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         IncorrectConfigTests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(inconfigs)),
                         IncorrectConfigTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         IncorrectConfigAPITests,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(inconfigs)),
                         IncorrectConfigAPITests::getTestCaseName);
}  // namespace
