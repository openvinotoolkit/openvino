// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/version.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> Multiconfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_MYRIAD}}
    };

    const std::vector<std::map<std::string, std::string>> Heteroconfigs = {
            {{ HETERO_CONFIG_KEY(DUMP_GRAPH_DOT) , CommonTestUtils::DEVICE_MYRIAD}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, VersionTest,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            VersionTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, VersionTest,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(Multiconfigs)),
                            VersionTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Hetero_BehaviorTests, VersionTest,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                    ::testing::ValuesIn(Heteroconfigs)),
                            VersionTest::getTestCaseName);


}  // namespace
