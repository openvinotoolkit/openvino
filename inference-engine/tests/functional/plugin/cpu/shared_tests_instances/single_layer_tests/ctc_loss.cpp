// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/ctc_loss.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> fPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};
const std::vector<InferenceEngine::Precision> iPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64
};

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
        ::testing::ValuesIn(unique)
);

INSTANTIATE_TEST_SUITE_P(smoke_Set1, CTCLossLayerTest,
                        ::testing::Combine(
                            ctcLossArgsSubset1,
                            ::testing::ValuesIn(fPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        CTCLossLayerTest::getTestCaseName);

const auto ctcLossArgsSubset2 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>({3, 6, 8})),                          // logits shape
        ::testing::ValuesIn(std::vector<std::vector<int>>({{6, 5, 6}, {5, 5, 5}})), // logits length
        ::testing::ValuesIn(std::vector<std::vector<std::vector<int>>>(
            {{{4, 1, 2, 3, 4, 5}, {5, 4, 3, 0, 1, 0}, {2, 1, 3, 1, 3, 0}},
             {{2, 1, 5, 3, 2, 6}, {3, 3, 3, 3, 3, 3}, {6, 5, 6, 5, 6, 5}}})),       // labels
        ::testing::ValuesIn(std::vector<std::vector<int>>({{4, 3, 5}, {3, 3, 5}})), // labels length
        ::testing::ValuesIn(std::vector<int>({0, 7})),                              // blank index
        ::testing::ValuesIn(preprocessCollapseRepeated),
        ::testing::ValuesIn(ctcMergeRepeated),
        ::testing::ValuesIn(unique)
);

INSTANTIATE_TEST_SUITE_P(smoke_Set2, CTCLossLayerTest,
                        ::testing::Combine(
                            ctcLossArgsSubset2,
                            ::testing::ValuesIn(fPrecisions),
                            ::testing::ValuesIn(iPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        CTCLossLayerTest::getTestCaseName);
}  // namespace
