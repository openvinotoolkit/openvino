// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<ov::test::InputShape>> originalShape_4D {
    { {{}, {{2, 12, 197, 64}}}, {{}, {{2, 12, 64, 197}}}, {{}, {{2, 12, 197, 64}}} },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 128, 64}, {1, 12, 197, 100}, {1, 3, 128, 64}, {1, 12, 197, 600}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 64, 128}, {1, 12, 100, 197}, {1, 3, 64, 128}, {1, 12, 600, 197}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 128, 64}, {1, 12, 197, 100}, {1, 3, 128, 64}, {1, 12, 197, 600}}},
    },
};

std::vector<std::vector<ov::test::InputShape>> originalShapes {
    { {{}, {{12, 197, 64}}},  {{}, {{12, 64, 197}}},  {{}, {{12, 197, 64}}} },
    { {{}, {{12, 128, 100}}}, {{}, {{12, 100, 128}}}, {{}, {{12, 128, 100}}} },
    { {{}, {{2, 12, 197, 64}}}, {{}, {{2, 12, 64, 197}}}, {{}, {{2, 12, 197, 64}}} },
    {
        {PartialShape{-1, -1, 64},  {{2, 9, 64},   {1, 64, 64},  {2, 64, 64}}},
        {PartialShape{-1, 64, 124}, {{2, 64, 124}, {1, 64, 124}, {2, 64, 124}}},
        {PartialShape{-1, 124, 64}, {{2, 124, 64}, {1, 124, 64}, {2, 124, 64}}},
    },
    {
        {PartialShape{2, -1, 64}, {{2, 9, 64}, {2, 4, 64}, {2, 9, 64}}},
        {PartialShape{2, 64, -1}, {{2, 64, 9}, {2, 64, 4}, {2, 64, 9}}},
        {PartialShape{2, -1, 64}, {{2, 9, 64}, {2, 4, 64}, {2, 9, 64}}},
    },
    {
        {PartialShape{-1, 128, 64}, {{1, 128, 64}, {2, 128, 64}, {1, 128, 64}}},
        {PartialShape{-1, 64, 128}, {{1, 64, 128}, {2, 64, 128}, {1, 64, 128}}},
        {PartialShape{-1, 128, 64}, {{1, 128, 64}, {2, 128, 64}, {1, 128, 64}}},
    },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 128, 64}, {1, 12, 197, 100}, {1, 3, 128, 64}, {1, 12, 197, 600}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 64, 128}, {1, 12, 100, 197}, {1, 3, 64, 128}, {1, 12, 600, 197}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 128, 64}, {1, 12, 197, 100}, {1, 3, 128, 64}, {1, 12, 197, 600}}},
    },
    {
        {PartialShape{1, 4, -1, -1}, {{1, 4, 384, 64}, {1, 4, 197, 64}, {1, 4, 384, 560}}},
        {PartialShape{1, 4, -1, -1}, {{1, 4, 64, 128}, {1, 4, 64, 197}, {1, 4, 560, 384}}},
        {PartialShape{1, 4, -1, 64}, {{1, 4, 128, 64}, {1, 4, 197, 64}, {1, 4, 384, 64}}},
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeOnInputs_4D,
    MHAWOTransposeOnInputs,
    ::testing::Combine(::testing::ValuesIn(originalShape_4D),
                       ::testing::Values(std::vector<ov::element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(true),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeFP32,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(originalShapes),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(true),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> originalShape_4D_fa {
    { {{}, {{2, 2, 1024, 256}}}, {{}, {{2, 2, 256, 6400}}}, {{}, {{2, 2, 6400, 256}}} },
    { {{}, {{1, 1, 1028, 256}}}, {{}, {{1, 1, 256, 6400}}}, {{}, {{1, 1, 6400, 256}}} },
    { {{}, {{1, 1, 1024, 256}}}, {{}, {{1, 1, 256, 6410}}}, {{}, {{1, 1, 6410, 256}}} },
    { {{}, {{1, 1, 1028, 256}}}, {{}, {{1, 1, 256, 6410}}}, {{}, {{1, 1, 6410, 256}}} },
    { {{}, {{1, 1, 1028, 128}}}, {{}, {{1, 1, 128, 6410}}}, {{}, {{1, 1, 6410, 128}}} },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 4096, 256}, {1, 1, 256, 4096}, {1, 1, 4096, 256}, {1, 1, 4096, 256}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 4100, 256}, {1, 1, 256, 8200}, {1, 1, 12300, 256}, {1, 1, 28736, 256}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 4100, 256}, {1, 1, 256, 8200}, {1, 1, 12300, 256}, {1, 1, 28736, 256}}},
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeFP32_fa,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(originalShape_4D_fa),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeBF16,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(originalShapes),
                       ::testing::ValuesIn(precision_bf16_if_supported(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(true),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeEnforceBF16,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(originalShapes),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::bf16),
                       ::testing::Values(true),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::cpu_bf16_plugin_config)),
    MHA::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeEnforceFP16,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(originalShapes),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::f16),
                       ::testing::Values(true),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::cpu_f16_plugin_config)),
    MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
