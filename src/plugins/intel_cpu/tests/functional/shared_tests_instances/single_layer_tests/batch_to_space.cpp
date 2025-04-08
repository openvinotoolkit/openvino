// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/batch_to_space.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::BatchToSpaceLayerTest;

const std::vector<ov::element::Type> net_types = {
        ov::element::f32,
        ov::element::i32
};

const std::vector<std::vector<ov::Shape>> data_shapes_2D_static = {
        {{12, 4}},
        {{48, 3}}
};

const std::vector<std::vector<int64_t>> block_shapes_2D = {
        {1, 2},
        {1, 6}
};

const std::vector<std::vector<int64_t>> crops_2D = {
        {0, 0},
        {0, 1}
};

const auto batch_to_space_2d_tests = ::testing::Combine(
        ::testing::ValuesIn(block_shapes_2D),
        ::testing::ValuesIn(crops_2D),
        ::testing::ValuesIn(crops_2D),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(data_shapes_2D_static)),
        ::testing::ValuesIn(net_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchToSpace_2D,
        BatchToSpaceLayerTest,
        batch_to_space_2d_tests,
        BatchToSpaceLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> data_shapes_4D_static = {
        {{4, 1, 2, 2}},
        {{4, 3, 2, 2}},
        {{8, 1, 3, 2}}
};

const std::vector<std::vector<int64_t>> block_shapes_4D = {
        {1, 1, 2, 2},
        {1, 2, 1, 2}
};

const std::vector<std::vector<int64_t>> crops_begin_4D = {
        {0, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 0, 2, 0}
};

const std::vector<std::vector<int64_t>> crops_end_4D = {
        {0, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 2}
};

const auto batch_to_space_4d_spatial_dims_tests = ::testing::Combine(
        ::testing::Values(block_shapes_4D[0]),
        ::testing::ValuesIn(crops_begin_4D),
        ::testing::ValuesIn(crops_end_4D),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(data_shapes_4D_static)),
        ::testing::ValuesIn(net_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto batch_to_space_4d_channel_dim_tests = ::testing::Combine(
        ::testing::Values(block_shapes_4D[1]),
        ::testing::Values(crops_begin_4D[0]),
        ::testing::Values(crops_end_4D[0]),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(data_shapes_4D_static)),
        ::testing::ValuesIn(net_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchToSpace_4D_spatial_dims,
        BatchToSpaceLayerTest,
        batch_to_space_4d_spatial_dims_tests,
        BatchToSpaceLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchToSpace_4D_channel_dim,
        BatchToSpaceLayerTest,
        batch_to_space_4d_channel_dim_tests,
        BatchToSpaceLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> data_shapes_5D_static = {
        {{12, 1, 2, 1, 2}}
};

const std::vector<std::vector<int64_t>> block_shapes_5D = {
        {1, 1, 3, 2, 2},
        {1, 2, 1, 2, 3}
};

const std::vector<std::vector<int64_t>> crops_begin_5D = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0}
};

const std::vector<std::vector<int64_t>> crops_end_5D = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 1},
        {0, 0, 0, 0, 1}
};

const auto batch_to_space_5d_spatial_dims_tests = ::testing::Combine(
        ::testing::Values(block_shapes_5D[0]),
        ::testing::ValuesIn(crops_begin_5D),
        ::testing::ValuesIn(crops_end_5D),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(data_shapes_5D_static)),
        ::testing::ValuesIn(net_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU));

const auto batch_to_space_5d_channel_dim_tests = ::testing::Combine(
        ::testing::Values(block_shapes_5D[1]),
        ::testing::Values(crops_begin_5D[0]),
        ::testing::Values(crops_end_5D[0]),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(data_shapes_5D_static)),
        ::testing::ValuesIn(net_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchToSpace_5D_spatial_dims,
        BatchToSpaceLayerTest,
        batch_to_space_5d_spatial_dims_tests,
        BatchToSpaceLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_BatchToSpace_5D_channel_dim,
        BatchToSpaceLayerTest,
        batch_to_space_5d_channel_dim_tests,
        BatchToSpaceLayerTest::getTestCaseName);

}  // namespace
