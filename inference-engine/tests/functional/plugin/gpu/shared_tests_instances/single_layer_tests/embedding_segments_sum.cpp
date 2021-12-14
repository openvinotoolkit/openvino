// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/embedding_segments_sum.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<InferenceEngine::Precision> indPrecisions = {
    InferenceEngine::Precision::I64,
    InferenceEngine::Precision::I32
};

const std::vector<std::vector<size_t>> emb_table_shape = {
    {5, 6},
    {10, 35},
    {5, 4, 16}
};
const std::vector<std::vector<size_t>> indices = {
    {0, 1, 2, 2, 3},
    {4, 4, 3, 1, 2}
};
const std::vector<std::vector<size_t>> segment_ids = {
    {0, 1, 2, 3, 4},
    {0, 0, 2, 2, 4}
};
const std::vector<size_t> num_segments = {5, 7};
const std::vector<size_t> default_index = {0, 4};
const std::vector<bool> with_weights = {false, true};
const std::vector<bool> with_default_index = {false, true};

const auto embSegmentsSumArgSet = ::testing::Combine(
    ::testing::ValuesIn(emb_table_shape), ::testing::ValuesIn(indices),
    ::testing::ValuesIn(segment_ids), ::testing::ValuesIn(num_segments),
    ::testing::ValuesIn(default_index), ::testing::ValuesIn(with_weights),
    ::testing::ValuesIn(with_default_index));

INSTANTIATE_TEST_SUITE_P(
    smoke, EmbeddingSegmentsSumLayerTest,
    ::testing::Combine(embSegmentsSumArgSet, ::testing::ValuesIn(netPrecisions),
                       ::testing::ValuesIn(indPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    EmbeddingSegmentsSumLayerTest::getTestCaseName);
}  // namespace
