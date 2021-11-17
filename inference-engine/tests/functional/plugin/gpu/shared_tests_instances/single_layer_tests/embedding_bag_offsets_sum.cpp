// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/embedding_bag_offsets_sum.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::test::ElementType::f32,
    ov::test::ElementType::f16
};

const std::vector<ov::test::ElementType> indPrecisions = {
    ov::test::ElementType::i64,
    ov::test::ElementType::i32
};

const std::vector<ov::test::InputShape> input_shapes = {
    {{5, 6}, {{5, 6}}},
    {{10, 35}, {{10, 35}}},
    {{5, 4, 16}, {{5, 4, 16}}},
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
    ::testing::ValuesIn(input_shapes), ::testing::ValuesIn(indices),
    ::testing::ValuesIn(offsets), ::testing::ValuesIn(default_index),
    ::testing::ValuesIn(with_weights), ::testing::ValuesIn(with_default_index));

INSTANTIATE_TEST_SUITE_P(
    smoke, EmbeddingBagOffsetsSumLayerTest,
    ::testing::Combine(embBagOffsetSumArgSet,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::ValuesIn(indPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    EmbeddingBagOffsetsSumLayerTest::getTestCaseName);
}  // namespace
