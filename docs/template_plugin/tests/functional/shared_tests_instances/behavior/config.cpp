// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"

#include "behavior/config.hpp"

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values("TEMPLATE"),
                                    ::testing::ValuesIn(configs)),
                            IncorrectConfigTests::getTestCaseName);


    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, IncorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values("TEMPLATE"),
                                    ::testing::ValuesIn(configs)),
                            IncorrectConfigAPITests::getTestCaseName);


    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, CorrectConfigAPITests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values("TEMPLATE"),
                                    ::testing::ValuesIn(configs)),
                            CorrectConfigAPITests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, CorrectConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values("TEMPLATE"),
                                    ::testing::ValuesIn(configs)),
                            CorrectConfigAPITests::getTestCaseName);
} // namespace