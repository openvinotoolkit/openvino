// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {
    {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         ExecutableNetworkBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ExecutableNetworkBaseTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> setPRC = {InferenceEngine::Precision::FP32,
                                                        InferenceEngine::Precision::U8,
                                                        InferenceEngine::Precision::I16};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         ExecNetSetPrecision,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ExecNetSetPrecision::getTestCaseName);

}  // namespace
