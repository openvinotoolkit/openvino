// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior2/infer_request/io_blob.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
        ::testing::Combine(
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                ::testing::Values(std::map<std::string, std::string>({}))),
        InferRequestIOBBlobTest::getTestCaseName);


}  // namespace
