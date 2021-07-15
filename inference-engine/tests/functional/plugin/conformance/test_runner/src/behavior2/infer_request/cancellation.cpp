// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior2/infer_request/cancellation.hpp"
#include "conformance.hpp"

namespace ConformanceTests {
using namespace BehaviorTestsDefinitions;


namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestCancellationTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(targetDevice),
                                 ::testing::ValuesIn(configs)),
                         InferRequestCancellationTests::getTestCaseName);
}  // namespace
}  // namespace ConformanceTests
