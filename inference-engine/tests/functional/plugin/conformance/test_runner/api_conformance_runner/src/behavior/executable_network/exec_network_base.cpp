// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"

#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(ConformanceTests::targetDevice),
                                    ::testing::ValuesIn(configs)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO))),
                            ExecutableNetworkBaseTest::getTestCaseName);

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::I16,
            InferenceEngine::Precision::U16
    };

    const std::vector<std::map<std::string, std::string>> configSetPrc = {
            {},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(ConformanceTests::targetDevice),
                                    ::testing::ValuesIn(configSetPrc)),
                            ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_MULTI))),
                            ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(generateConfigs(CommonTestUtils::DEVICE_AUTO))),
                            ExecNetSetPrecision::getTestCaseName);
}  // namespace
