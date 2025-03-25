// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/tile.hpp"

using ov::test::TileLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {
        ov::element::i8,
        ov::element::u8,
        ov::element::i32,
        ov::element::bf16,
        ov::element::f32
};

const std::vector<ov::element::Type> model_t_types = {
        ov::element::i64,
        ov::element::i16,
        ov::element::f16,
};

const std::vector<std::vector<ov::Shape>> input_shape_static_3D = {
        {{2, 3, 4}},
        {{1, 1, 1}},
};

const std::vector<std::vector<int64_t>> repeats_3D = {
        {1, 2, 3},
        {1, 1, 2, 3},
        {1, 2, 1, 3},
        {2, 1, 1},
        {2, 3, 1},
        {2, 2, 2},
        {1, 1, 1}
};

INSTANTIATE_TEST_SUITE_P(smoke_Tile, TileLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(repeats_3D),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_3D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        TileLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PrecTransformation, TileLayerTest,
        ::testing::Combine(
                ::testing::Values(repeats_3D[0]),
                ::testing::ValuesIn(model_t_types),
                ::testing::Values(ov::test::static_shapes_to_test_representation(input_shape_static_3D)[0]),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        TileLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> repeats_6D = {
        {1, 1, 1, 2, 1, 2},
        {1, 1, 1, 1, 1, 1}
};

const std::vector<std::vector<ov::Shape>> input_shape_static_6D = {
        {{1, 4, 3, 1, 3, 1}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Tile6d, TileLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(repeats_6D),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static_6D)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        TileLayerTest::getTestCaseName);

}  // namespace
