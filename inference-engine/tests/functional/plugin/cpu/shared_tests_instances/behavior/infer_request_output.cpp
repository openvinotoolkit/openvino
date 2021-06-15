// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_output.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
            {{InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };

    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}}
    };

    const std::vector<std::map<std::string, std::string>> autoConfigs = {
            {{InferenceEngine::KEY_AUTO_DEVICE_LIST , CommonTestUtils::DEVICE_CPU}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferRequestOutputTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                    ::testing::ValuesIn(configs)),
                            InferRequestOutputTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, InferRequestOutputTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            InferRequestOutputTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Auto_BehaviorTests, InferRequestOutputTests,
                            ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(autoConfigs)),
                            InferRequestOutputTests::getTestCaseName);

}  // namespace
