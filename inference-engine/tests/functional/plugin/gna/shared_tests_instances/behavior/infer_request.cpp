// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request.hpp"
#include "ie_plugin_config.hpp"
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferRequestTests,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                ::testing::Values(std::map<std::string, std::string>({}))),
                InferRequestTests::getTestCaseName);

}  // namespace
