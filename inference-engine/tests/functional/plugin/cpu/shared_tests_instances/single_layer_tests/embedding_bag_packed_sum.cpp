// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/embedding_bag_packed_sum.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> indPrecisions = {
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I32
};

const std::vector<std::vector<size_t>> emb_table_shape = {{5, 6}, {10, 35}, {5, 4, 16}};
const std::vector<std::vector<std::vector<size_t>>> indices =
        {{{0, 1}, {2, 2}, {3, 4}}, {{4, 4, 3}, {1, 0, 2}}, {{1, 2, 1, 2}, {1, 2, 1, 2}}};
const std::vector<bool> with_weights = {false, true};

const auto embBagPackedSumArgSet = ::testing::Combine(
        ::testing::ValuesIn(emb_table_shape),
        ::testing::ValuesIn(indices),
        ::testing::ValuesIn(with_weights)
);

INSTANTIATE_TEST_SUITE_P(smoke, EmbeddingBagPackedSumLayerTest,
                        ::testing::Combine(
                                embBagPackedSumArgSet,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(indPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        EmbeddingBagPackedSumLayerTest::getTestCaseName);
}  // namespace
