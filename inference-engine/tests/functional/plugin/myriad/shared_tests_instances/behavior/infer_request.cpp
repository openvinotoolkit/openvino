// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request.hpp"
#include "ie_plugin_config.hpp"
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferRequestTests,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                ::testing::Values(std::map<std::string, std::string>({}))),
                InferRequestTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, InferRequestTests,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                ::testing::Values(std::map<std::string, std::string>
                ({{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD}}))),
                InferRequestTests::getTestCaseName);
}  // namespace
