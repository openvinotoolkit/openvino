// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/embedding_segments_sum.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(EmbeddingSegmentsSumLayerTest, Serialize) {
        serialize();
    }

    const std::vector<ov::test::ElementType> netPrecisions = {
        ov::test::ElementType::f32,
        ov::test::ElementType::i32,
        ov::test::ElementType::u8};

    const std::vector<ov::test::ElementType> indPrecisions = {
        ov::test::ElementType::i64,
        ov::test::ElementType::i32};

    const std::vector<ov::test::InputShape> input_shapes = {
        {{5, 6}, {{5, 6}}},
        {{5, 4, 16}, {{5, 4, 16}}}
    };

    const std::vector<std::vector<size_t>> indices =
        {{0, 1, 2, 2, 3}, {4, 4, 3, 1, 2}};
    const std::vector<std::vector<size_t>> segment_ids = {{0, 1, 2, 3, 4}, {0, 0, 2, 2, 4}};
    const std::vector<size_t> num_segments = {5, 7};
    const std::vector<size_t> default_index = {0, 4};
    const std::vector<bool> with_weights = {false, true};
    const std::vector<bool> with_default_index = {false, true};

    const auto EmbeddingSegmentsSumParams = ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(indices),
        ::testing::ValuesIn(segment_ids),
        ::testing::ValuesIn(num_segments),
        ::testing::ValuesIn(default_index),
        ::testing::ValuesIn(with_weights),
        ::testing::ValuesIn(with_default_index));

    INSTANTIATE_TEST_SUITE_P(
        smoke_EmbeddingSegmentsSumLayerTest_Serialization, EmbeddingSegmentsSumLayerTest,
        ::testing::Combine(EmbeddingSegmentsSumParams,
                           ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(indPrecisions),
                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        EmbeddingSegmentsSumLayerTest::getTestCaseName);
} // namespace
