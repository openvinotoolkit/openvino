// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/exec_layers.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {},
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, ExecLayerTests,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs)),
            ExecLayerTests::getTestCaseName);

}  // namespace
