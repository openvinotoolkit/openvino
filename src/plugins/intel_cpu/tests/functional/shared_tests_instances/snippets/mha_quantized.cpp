// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> inputShapesQuantized {
    {
        {{}, {{1, 128, 16, 64}}},
        {{}, {{1, 128, 16, 64}}},
        {{}, {{1, 16, 1, 1}}},
        {{}, {{1, 128, 16, 64}}}
    },
    {
        {{}, {{2, 68, 6, 92}}},
        {{}, {{2, 68, 6, 92}}},
        {{}, {{1, 1, 68, 68}}},
        {{}, {{2, 68, 6, 92}}}
    },
    // K, N are static
    {
        {PartialShape{-1, -1, -1, 100},  {{1, 64, 4, 100},  {2, 16, 2, 100},  {1, 72, 4, 100}}},
        {PartialShape{-1, 128, -1, 100}, {{1, 128, 4, 100}, {2, 128, 2, 100}, {1, 128, 4, 100}}},
        {PartialShape{-1, -1, -1, 128},  {{1, 4, 64, 128},  {2, 2, 16, 128},  {1, 4, 72, 128}}},
        {PartialShape{-1, 128, -1, 100}, {{1, 128, 4, 100}, {2, 128, 2, 100}, {1, 128, 4, 100}}},
    },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64},  {2, 16, 2, 100},  {1, 128, 3, 64},  {1, 128, 12, 600}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 1, 64},  {2, 128, 2, 100}, {1, 128, 1, 64},  {1, 128, 12, 600}}},
        {PartialShape{-1, -1, -1, -1}, {{2, 1, 128, 128}, {1, 1, 1, 128},   {2, 1, 128, 128}, {1, 12, 1, 1}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64},  {2, 128, 2, 100}, {1, 128, 3, 64},  {1, 128, 12, 600}}},
    }
};

// RISC-V keeps FakeQuantize in f32 with low-precision transformations disabled.
#if defined(OPENVINO_ARCH_RISCV64)
static constexpr size_t expected_nodes_mha_int8 = 6;           // FQx3, Transpose1 on inputs + MHA + Transpose on output
static constexpr size_t expected_subgraphs_mha_int8 = 4;       // FQx3 on inputs + MHA
static constexpr size_t expected_nodes_mha_quant_matmul0 = 5;  // FQx2, Transpose1 on inputs + MHA + Transpose on output
static constexpr size_t expected_subgraphs_mha_quant_matmul0 = 3;    // FQx2 on inputs + MHA
static constexpr size_t expected_nodes_mha_fq_after_matmul = 3;      // Transpose1 + MHA + Transpose on output
static constexpr size_t expected_subgraphs_mha_fq_after_matmul = 2;  // Transpose1 + MHA
static constexpr size_t expected_nodes_mha_fq = 5;                   // Transposex2 + Subgraphsx3
static constexpr size_t expected_subgraphs_mha_fq = 3;               // MHA + 2 x FQ on inputs
#else
static constexpr size_t expected_nodes_mha_int8 =
    7;  // FQx3, Transpose1 on inputs + MHA + Transpose on output + Deq Mul
static constexpr size_t expected_subgraphs_mha_int8 = 5;  // FQx3 on inputs + MHA + Deq Mul
static constexpr size_t expected_nodes_mha_quant_matmul0 =
    6;  // FQx2, Transpose1 on inputs + MHA + Transpose on output + Deq Mul
static constexpr size_t expected_subgraphs_mha_quant_matmul0 = 4;    // FQx2 on inputs + MHA + Deq Mul
static constexpr size_t expected_nodes_mha_fq_after_matmul = 4;      // Transpose1 + MHA + Transpose on output + Deq Mul
static constexpr size_t expected_subgraphs_mha_fq_after_matmul = 3;  // Transpose1 + MHA + Deq Mul
static constexpr size_t expected_nodes_mha_fq = 8;                   // Transposex3 + Subgraphsx5
static constexpr size_t expected_subgraphs_mha_fq = 5;  // MHA + Deq Mul on output + Deqs on inputs + 2 xFQ on inputs
#endif

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAINT8MatMul,
    MHAINT8MatMul,
    ::testing::Combine(::testing::ValuesIn(inputShapesQuantized),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),  // The graph doesn't contain Multiply
                       ::testing::Values(expected_nodes_mha_int8),
                       ::testing::Values(expected_subgraphs_mha_int8),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAQuantMatMul0,
    MHAQuantMatMul0,
    ::testing::Combine(
        ::testing::ValuesIn(inputShapesQuantized),
        ::testing::Values(std::vector<element::Type>{}),
        ::testing::Values(ov::element::f32),
        ::testing::Values(false),  // The graph doesn't contain Multiply
        ::testing::Values(expected_nodes_mha_quant_matmul0),
        ::testing::Values(expected_subgraphs_mha_quant_matmul0),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAFQAfterMatMul_4D,
    MHAFQAfterMatMul,
    ::testing::Combine(::testing::ValuesIn(inputShapesQuantized),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),  // The graph doesn't contain Multiply
                       ::testing::Values(expected_nodes_mha_fq_after_matmul),
                       ::testing::Values(expected_subgraphs_mha_fq_after_matmul),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAFQ,
    MHAFQ,
    ::testing::Combine(::testing::ValuesIn(inputShapesQuantized),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),  // The graph doesn't contain Multiply
                       ::testing::Values(expected_nodes_mha_fq),
                       ::testing::Values(expected_subgraphs_mha_fq),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
