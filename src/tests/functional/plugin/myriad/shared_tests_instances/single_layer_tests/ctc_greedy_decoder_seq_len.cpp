// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/ctc_greedy_decoder_seq_len.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> probPrecisions = {
    InferenceEngine::Precision::FP16
};
const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32
};

std::vector<bool> mergeRepeated{true, false};

const auto inputShape = std::vector<std::vector<size_t>>{
    {1, 1, 1}, {1, 6, 10}, {5, 3, 55},
    {4, 80, 80}, {80, 4, 80}, {80, 80, 4}, {8, 20, 128}
};

const auto sequenceLengths = std::vector<int>{1, 10, 50, 100};

const auto blankIndexes = std::vector<int>{0, 10, 100};

INSTANTIATE_TEST_SUITE_P(smoke, CTCGreedyDecoderSeqLenLayerTest,
        ::testing::Combine(
                        ::testing::ValuesIn(inputShape),
                        ::testing::ValuesIn(sequenceLengths),
                        ::testing::ValuesIn(probPrecisions),
                        ::testing::ValuesIn(idxPrecisions),
                        ::testing::ValuesIn(blankIndexes),
                        ::testing::ValuesIn(mergeRepeated),
                        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                    CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);
}  // namespace
