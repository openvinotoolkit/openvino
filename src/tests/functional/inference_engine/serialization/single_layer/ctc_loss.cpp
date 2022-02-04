// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/ctc_loss.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(CTCLossLayerTest, Serialize) { Serialize(); }

const std::vector<InferenceEngine::Precision> fPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16};
const std::vector<InferenceEngine::Precision> iPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::I64};

const std::vector<bool> preprocessCollapseRepeated = {true, false};
const std::vector<bool> ctcMergeRepeated = {true, false};
const std::vector<bool> unique = {true, false};

const auto ctcLossArgsSubset1 = ::testing::Combine(
    ::testing::Values(std::vector<size_t>({2, 3, 3})),                    // logits shape
    ::testing::ValuesIn(std::vector<std::vector<int>>({{2, 3}, {3, 3}})), // logits length
    ::testing::ValuesIn(std::vector<std::vector<std::vector<int>>>(
        {{{0, 1, 0}, {1, 0, 1}}, {{0, 1, 2}, {1, 1, 1}}})),               // labels
    ::testing::ValuesIn(std::vector<std::vector<int>>({{2, 2}, {2, 1}})), // labels length
    ::testing::Values(2),                                                 // blank index
    ::testing::ValuesIn(preprocessCollapseRepeated),
    ::testing::ValuesIn(ctcMergeRepeated),
    ::testing::ValuesIn(unique));

INSTANTIATE_TEST_SUITE_P(smoke_CTCLossSerialization, CTCLossLayerTest,
                            ::testing::Combine(
                                ctcLossArgsSubset1,
                                ::testing::ValuesIn(fPrecisions),
                                ::testing::ValuesIn(iPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            CTCLossLayerTest::getTestCaseName);
} // namespace
