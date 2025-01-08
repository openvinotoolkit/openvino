// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<ov::test::InputShape>> transposedShape_4D_WithMul {
    {
        {PartialShape{-1, -1, -1, 100},  {{1, 64, 4, 100},  {2, 16, 2, 100},  {1, 72, 4, 100}}},
        {PartialShape{-1, 200, -1, 100}, {{1, 200, 4, 100}, {2, 200, 2, 100}, {1, 200, 4, 100}}},
        {PartialShape{-1, -1, 100, 200}, {{1, 4, 100, 200}, {2, 2, 100, 200}, {1, 4, 100, 200}}},
        {PartialShape{-1, -1, -1, 200},  {{1, 4, 64, 200},  {2, 2, 16, 200},  {1, 4, 72, 200}}},
        {PartialShape{-1, 200, -1, 100}, {{1, 200, 4, 100}, {2, 200, 2, 100}, {1, 200, 4, 100}}},
    },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {1, 70, 3, 19}, {1, 128, 3, 64}, {1, 68, 6, 87}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 1, 64}, {2, 49, 1, 19}, {1, 128, 1, 64}, {2, 13, 6, 87}}},
        {PartialShape{1},              {{1},             {1},            {1},             {1} }},
        {PartialShape{-1, -1, -1, -1}, {{2, 1, 128, 128}, {1, 1, 70, 49}, {2, 1, 128, 128}, {1, 1, 68, 13}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {1, 49, 3, 19}, {1, 128, 3, 64}, {2, 13, 6, 87}}},
    },
    {
        {PartialShape{-1, -1, 12, 64}, {{1, 70, 12, 64}, {1, 20, 12, 64}, {1, 20, 12, 64}, {1, 20, 12, 64}, {1, 70, 12, 64}}},
        {PartialShape{-1, -1, 12, 64}, {{1, 35, 12, 64}, {2, 10, 12, 64}, {2, 1, 12, 64},  {2, 10, 12, 64}, {1, 35, 12, 64}}},
        {PartialShape{-1, 12, 64, -1}, {{1, 12, 64, 35}, {1, 12, 64, 10}, {1, 12, 64, 10}, {1, 12, 64, 1},  {1, 12, 64, 35}}},
        {PartialShape{-1, 12, -1, -1}, {{2, 12, 70, 35}, {1, 12, 20, 10}, {1, 12, 20, 10}, {1, 12, 20, 1},  {2, 12, 70, 35}}},
        {PartialShape{-1, -1, 12, 64}, {{1, 35, 12, 64}, {1, 10, 12, 64}, {1, 10, 12, 64}, {1, 10, 12, 64}, {1, 35, 12, 64}}},
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_4D_WithDynamicMul,
    MHAWithDynamicMul,
    ::testing::Combine(::testing::ValuesIn(transposedShape_4D_WithMul),
                       ::testing::ValuesIn(precision_f32(5)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(2), // Transpose1 + MHA
                       ::testing::Values(2), // Transpose1 + MHA
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHAWithDynamicMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_4D_WithDynamicMul_EnforceBF16,
    MHAWithDynamicMul,
    ::testing::Combine(::testing::ValuesIn(transposedShape_4D_WithMul),
                       ::testing::ValuesIn(precision_f32(5)),
                       ::testing::Values(ov::element::bf16),
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(9),  // Transpose1 + MHA + 1 Transpose on output + 6 Converts around
                       ::testing::Values(7),  // MHA + 6 Converts around
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHAWithDynamicMul::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
