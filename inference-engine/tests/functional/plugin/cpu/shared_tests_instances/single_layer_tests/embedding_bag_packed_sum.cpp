// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/embedding_bag_packed_sum.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<ov::test::ElementType> netPrecisions = {
        ov::test::ElementType::f32,
        ov::test::ElementType::i32,
        ov::test::ElementType::u8
};

const std::vector<ov::test::ElementType> indPrecisions = {
        ov::test::ElementType::i64,
        ov::test::ElementType::i32
};

const std::vector<ov::test::InputShape> input_shapes = {
        // dynamic input shapes
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{{5, 6}}, {10, 35}}
        },
        {
            // input model dynamic shapes
            {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
            // input tensor shapes
            {{5, 4, 16}, {10, 12, 8}}
        },
        {
            // input model dynamic shapes with limits
            {{5, 10}, {6, 35}, {4, 8}},
            // input tensor shapes
            {{5, 6, 4}, {10, 35, 8}, {5, 6, 4}}
        },
        // static shapes
        {{5, 6}, {{5, 6}}},
        {{10, 35}, {{10, 35}}},
        {{5, 4, 16}, {{5, 4, 16}}},
};

const std::vector<std::vector<std::vector<size_t>>> indices =
        {{{0, 1}, {2, 2}, {3, 4}}, {{4, 4, 3}, {1, 0, 2}}, {{1, 2, 1, 2}, {1, 2, 1, 2}}};
const std::vector<bool> with_weights = {false, true};

const auto embBagPackedSumArgSet = ::testing::Combine(
        ::testing::ValuesIn(input_shapes),
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
