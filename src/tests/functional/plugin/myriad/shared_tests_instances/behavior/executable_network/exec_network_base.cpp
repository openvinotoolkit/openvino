// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> MultiConfigsInputOutput = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , CommonTestUtils::DEVICE_MYRIAD}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                             ::testing::Combine(
                                     ::testing::ValuesIn(netPrecisions),
                                     ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                     ::testing::ValuesIn(configs)),
                             ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecNetSetPrecision,
                             ::testing::Combine(
                                     ::testing::ValuesIn(netPrecisions),
                                     ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                     ::testing::ValuesIn(MultiConfigsInputOutput)),
                             ExecNetSetPrecision::getTestCaseName);
}  // namespace