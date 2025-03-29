// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/tile.hpp"

namespace {
using ov::test::TileLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<std::vector<int64_t>> repeats = {
        {2, 3},
        {1, 2, 3},
        {2, 1, 1},
        {2, 3, 1},
        {2, 2, 2},
        {2, 3, 4, 5},
};

INSTANTIATE_TEST_SUITE_P(smoke_Tile, TileLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(repeats),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{2, 3, 4}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        TileLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Tile6d, TileLayerTest,
        ::testing::Combine(
                ::testing::Values(std::vector<int64_t>({1, 1, 1, 2, 1, 2})),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 4, 3, 1, 3, 1}}))),
                ::testing::Values(ov::test::utils::DEVICE_GPU)),
        TileLayerTest::getTestCaseName);

}  // namespace
