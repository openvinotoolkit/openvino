// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

static ov::AnyMap set_num_threads(size_t num_threads) {
    return ov::AnyMap({ov::inference_num_threads(num_threads)});
}

std::vector<std::vector<ov::test::InputShape>> splitm_shapes_4d = {
    {
        {PartialShape{}, {{1, 128, 2, 64}}},
        {PartialShape{}, {{1, 128, 2, 64}}},
        {PartialShape{}, {{1, 1, 1, 1}}},
        {PartialShape{}, {{1, 128, 2, 64}}},
    },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 2, 64}, {1, 17, 2, 64}, {1, 128, 2, 64}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 2, 64}, {1, 17, 2, 64}, {1, 128, 2, 64}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 1, 128}, {1, 1, 1, 17}, {1, 1, 1, 128}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 2, 64}, {1, 17, 2, 64}, {1, 128, 2, 64}}},
    },
    {
        {PartialShape{-1, 128, -1, -1}, {{1, 128, 2, 64}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 16, 2, 64}}},
        {PartialShape{-1, -1, 128, -1}, {{1, 1, 128, 16}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 16, 2, 32}}},
    },
    {
        {PartialShape{-1, 32, -1, -1}, {{1, 32, 2, 64}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 16, 2, 64}}},
        {PartialShape{-1, -1, 32, -1}, {{1, 1, 32, 16}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 16, 2, 32}}},
    },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 16, 2, 64}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 16, 2, 64}}},
        {PartialShape{-1, -1, 16, -1}, {{1, 1, 16, 16}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 16, 2, 32}}},
    },
};

static constexpr size_t expected_nodes_mha_splitm_4d_dyn = 2;

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_4D_SplitDimensionM,
    MHAWithThreadCount,
    ::testing::Combine(::testing::ValuesIn(splitm_shapes_4d),
                       ::testing::ValuesIn(precision_f32(4)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
                       ::testing::Values(expected_nodes_mha_splitm_4d_dyn),
                       ::testing::Values(2),  // Transpose1 + MHA
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(set_num_threads(4))),
    MHAWithThreadCount::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> splitm_shapes_3d = {
    {
        {PartialShape{}, {{384, 2, 64}}},
        {PartialShape{}, {{384, 2, 64}}},
        {PartialShape{}, {{1, 384, 384}}},
        {PartialShape{}, {{384, 2, 64}}},
    },
    {
        {PartialShape{-1, -1, -1}, {{128, 2, 64}, {17, 2, 64}, {128, 2, 64}}},
        {PartialShape{-1, -1, -1}, {{128, 2, 64}, {17, 2, 64}, {128, 2, 64}}},
        {PartialShape{-1, -1, -1}, {{1, 1, 128}, {1, 1, 17}, {1, 1, 128}}},
        {PartialShape{-1, -1, -1}, {{128, 2, 64}, {17, 2, 64}, {128, 2, 64}}},
    },
    {
        {PartialShape{-1, 2, 64}, {{128, 2, 64}, {64, 2, 64}, {128, 2, 64}}},
        {PartialShape{-1, 2, 64}, {{128, 2, 64}, {64, 2, 64}, {128, 2, 64}}},
        {PartialShape{1, 1, -1}, {{1, 1, 128}, {1, 1, 64}, {1, 1, 128}}},
        {PartialShape{-1, 2, 64}, {{128, 2, 64}, {64, 2, 64}, {128, 2, 64}}},
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_3D_SplitDimensionM,
    MHAWithThreadCount,
    ::testing::Combine(::testing::ValuesIn(splitm_shapes_3d),
                       ::testing::ValuesIn(precision_f32(4)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
                       ::testing::Values(5),  // Subgraph + 4 Transpose
                       ::testing::Values(2),  // MHA + one of the transposes is executed via Subgraph (because callback is disabled)
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(set_num_threads(4))),
    MHAWithThreadCount::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
