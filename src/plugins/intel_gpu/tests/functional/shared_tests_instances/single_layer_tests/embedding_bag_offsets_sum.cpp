// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/embedding_bag_offsets_sum.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::EmbeddingBagOffsetsSumLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<ov::element::Type> indPrecisions = {
    ov::element::i64,
    ov::element::i32
};

const std::vector<std::vector<ov::Shape>> emb_table_shape = {
    {{5, 6}},
    {{10, 35}},
    {{5, 4, 16}}
};

const std::vector<std::vector<size_t>> indices = {
    {0, 1, 2, 2, 3},
    {4, 4, 3, 1, 0},
    {1, 2, 1, 2, 1, 2, 1, 2, 1, 2}
};

const std::vector<std::vector<size_t>> offsets = {
    {0, 2},
    {0, 0, 2, 2},
    {2, 4}
};

const std::vector<size_t> default_index = {0, 4};
const std::vector<bool> with_weights = {false, true};
const std::vector<bool> with_default_index = {false, true};

const auto embBagOffsetSumArgSet = ::testing::Combine(
    ::testing::ValuesIn(indices),
    ::testing::ValuesIn(offsets),
    ::testing::ValuesIn(default_index),
    ::testing::ValuesIn(with_weights),
    ::testing::ValuesIn(with_default_index));

INSTANTIATE_TEST_SUITE_P(
    smoke, EmbeddingBagOffsetsSumLayerTest,
    ::testing::Combine(embBagOffsetSumArgSet,
                       ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(emb_table_shape)),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::ValuesIn(indPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    EmbeddingBagOffsetsSumLayerTest::getTestCaseName);
}  // namespace
