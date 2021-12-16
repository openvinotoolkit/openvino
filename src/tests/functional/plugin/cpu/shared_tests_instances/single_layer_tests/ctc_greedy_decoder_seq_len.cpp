// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/ctc_greedy_decoder_seq_len.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

std::vector<std::vector<size_t>> inputShape{{1, 1, 1}, {1, 6, 10}, {3, 3, 16}, {5, 3, 55}};

const std::vector<InferenceEngine::Precision> probPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};
const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::I64
};

std::vector<bool> mergeRepeated{true, false};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(inputShape),
    ::testing::Values(10),
    ::testing::ValuesIn(probPrecisions),
    ::testing::ValuesIn(idxPrecisions),
    ::testing::Values(0),
    ::testing::ValuesIn(mergeRepeated),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_set1, CTCGreedyDecoderSeqLenLayerTest,
                        basicCases,
                        CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set2, CTCGreedyDecoderSeqLenLayerTest,
        ::testing::Combine(
                        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 8, 11}, {4, 10, 55}}),
                        ::testing::ValuesIn(std::vector<int>{5, 100}),
                        ::testing::ValuesIn(probPrecisions),
                        ::testing::ValuesIn(idxPrecisions),
                        ::testing::ValuesIn(std::vector<int>{0, 5, 10}),
                        ::testing::ValuesIn(mergeRepeated),
                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                    CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);
}  // namespace
