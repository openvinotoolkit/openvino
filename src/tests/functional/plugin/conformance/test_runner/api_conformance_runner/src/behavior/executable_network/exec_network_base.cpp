// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"

#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(ov::test::conformance::targetDevice),
                                    ::testing::ValuesIn(empty_config)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_MULTI))),
                            ExecutableNetworkBaseTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_AUTO))),
                            ExecutableNetworkBaseTest::getTestCaseName);

    const std::vector<InferenceEngine::Precision> execNetBaseElemTypes = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::I16,
            InferenceEngine::Precision::U16
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(execNetBaseElemTypes),
                                    ::testing::Values(ov::test::conformance::targetDevice),
                                    ::testing::ValuesIn(empty_config)),
                            ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(execNetBaseElemTypes),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_MULTI))),
                            ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(execNetBaseElemTypes),
                                    ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                    ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_AUTO))),
                            ExecNetSetPrecision::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, ExecNetSetPrecision,
                             ::testing::Combine(
                                     ::testing::ValuesIn(execNetBaseElemTypes),
                                     ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                     ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_HETERO))),
                             ExecNetSetPrecision::getTestCaseName);
}  // namespace
