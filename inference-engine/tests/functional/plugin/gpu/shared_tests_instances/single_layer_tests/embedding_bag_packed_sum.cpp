// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/embedding_bag_packed_sum.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::test::ElementType::f32,
    ov::test::ElementType::f16,
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

const std::vector<std::vector<std::vector<size_t>>> indices = {
    {{0, 1}, {2, 2}, {3, 4}},
    {{4, 4, 3}, {1, 0, 2}},
    {{1, 2, 1, 2}, {1, 2, 1, 2}}
};
const std::vector<bool> with_weights = {false, true};

const auto embBagPackedSumArgSet = ::testing::Combine(
    ::testing::ValuesIn(input_shapes), ::testing::ValuesIn(indices),
    ::testing::ValuesIn(with_weights));

INSTANTIATE_TEST_SUITE_P(
    smoke, EmbeddingBagPackedSumLayerTest,
    ::testing::Combine(embBagPackedSumArgSet,
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::ValuesIn(indPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU)),
    EmbeddingBagPackedSumLayerTest::getTestCaseName);
}  // namespace
