// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/embedding_bag_packed_sum.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(EmbeddingBagPackedSumLayerTest, Serialize) {
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
        {{10, 35}, {{10, 35}}},
        {{5, 4, 16}, {{5, 4, 16}}}
    };

    const std::vector<std::vector<std::vector<size_t>>> indices =
        {{{0, 1}, {2, 2}}, {{4, 4, 3}, {1, 0, 2}}, {{1, 2, 1, 2}}};
    const std::vector<bool> with_weights = {false, true};

    const auto EmbeddingBagPackedSumParams = ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(indices),
        ::testing::ValuesIn(with_weights));

    INSTANTIATE_TEST_SUITE_P(
        smoke_EmbeddingBagPackedSumLayerTest_Serialization, EmbeddingBagPackedSumLayerTest,
        ::testing::Combine(EmbeddingBagPackedSumParams,
                           ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(indPrecisions),
                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        EmbeddingBagPackedSumLayerTest::getTestCaseName);
} // namespace
