// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

static ov::AnyMap enable_callback() {
    return ov::AnyMap({ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::ENABLE)});
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_4D_SplitDimensionM_static,
    MHA,
    ::testing::Combine(::testing::ValuesIn(SNIPPETS_TESTS_STATIC_SHAPES({{1, 128, 2, 64}, {1, 128, 2, 64}, {1, 1, 1, 1}, {1, 128, 2, 64}})),
                       ::testing::ValuesIn(precision_f32(4)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(true),
                       ::testing::Values(4),  // 4 Threads
                       ::testing::Values(7),  // Subgraph + 4 Reshapes, Transpose1 on inputs and 1 Reshape on output
                       ::testing::Values(2),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(enable_callback())),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_3D_SplitDimensionM_static,
    MHA,
    ::testing::Combine(
        ::testing::ValuesIn(SNIPPETS_TESTS_STATIC_SHAPES({{384, 2, 64}, {384, 2, 64}, {1, 384, 384}, {384, 2, 64}})),
        ::testing::ValuesIn(precision_f32(4)),
        ::testing::Values(ov::element::f32),
        ::testing::Values(true),
        ::testing::Values(4),   // 4 Threads
        ::testing::Values(10),  // Subgraph + 4 Reshapes on inputs and 1 Reshape on output + 4 Transposes
        ::testing::Values(1),   // MHA
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(enable_callback())),
    MHA::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> splitm_dynamic_shapes_4d = {
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

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_4D_SplitDimensionM_dynamic,
    MHA,
    ::testing::Combine(::testing::ValuesIn(splitm_dynamic_shapes_4d),
                       ::testing::ValuesIn(precision_f32(4)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
                       ::testing::Values(4),  // 4 Threads
                       ::testing::Values(2), // Transpose1 + MHA
                       ::testing::Values(2), // Transpose1 + MHA
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> splitm_dynamic_shapes_3d = {
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
    smoke_Snippets_MHA_3D_SplitDimensionM_dynamic,
    MHA,
    ::testing::Combine(::testing::ValuesIn(splitm_dynamic_shapes_3d),
                       ::testing::ValuesIn(precision_f32(4)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
                       ::testing::Values(4),  // 4 Threads
                       ::testing::Values(5),  // Subgraph + 4 Transpose
                       ::testing::Values(2),  // MHA + one of the transposes is executed via Subgraph (because callback is disabled)
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
