// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

static inline std::vector<std::vector<element::Type>> precisions() {
    std::vector<std::vector<element::Type>> prc = precision_f32(2);
// Note: TPP doesn't support low precisions yet
#ifndef SNIPPETS_LIBXSMM_TPP
    auto quant = quantized_precisions_if_supported();
    std::copy(quant.begin(), quant.end(), std::back_inserter(prc));
    auto bfloat = precision_bf16_if_supported(2);
    std::copy(bfloat.begin(), bfloat.end(), std::back_inserter(prc));
#endif
    return prc;
}

std::vector<std::vector<ov::test::InputShape>> input_shapes{
    { {{}, {{2, 1, 3, 5}}},   {{}, {{1, 3, 5, 3}}} },
    { {{}, {{3, 1, 32, 14}}},   {{}, {{1, 3, 14, 37}}} },
    { {{}, {{1, 2, 37, 23}}},   {{}, {{2, 1, 23, 37}}} },
    { {{}, {{1, 1, 32, 23}}},   {{}, {{1, 1, 23, 68}}} },
    { {{}, {{1, 16, 384, 64}}},   {{}, {{1, 16, 64, 384}}} },
    { {{}, {{1, 1, 100, 700}}},   {{}, {{1, 1, 700, 100}}} },
    { {{}, {{1, 1, 100, 1024}}},   {{}, {{1, 1, 1024, 100}}} },
    { {{}, {{1, 1, 100, 2500}}},   {{}, {{1, 1, 2500, 100}}} },
    { {{}, {{1, 1, 100, 4500}}},   {{}, {{1, 1, 4500, 100}}} },
    // Only M dimension is dynamic + one one loop by M
    {
        {PartialShape{-1, 2, -1, 64}, {{2, 2, 64, 64}, {2, 2, 64, 64}, {2, 2, 35, 64},
                                       {2, 2, 120, 64}, {2, 2, 15, 64}, {2, 2, 35, 64}}},
        {PartialShape{-1, 2, 64, 32}, {{2, 2, 64, 32}, {2, 2, 64, 32}, {1, 2, 64, 32},
                                       {1, 2, 64, 32}, {2, 2, 64, 32}, {1, 2, 64, 32}}}
    },
    // Only M dimension is dynamic + all Loops (by M, N, K)
    {
        {PartialShape{2, 2, -1, 550}, {{2, 2, 64, 550}, {2, 2, 16, 550}, {2, 2, 35, 550},
                                       {2, 2, 16, 550}, {2, 2, 70, 550}, {2, 2, 64, 550}}},
        {PartialShape{2, 1, 550, 70}, {{2, 1, 550, 70}, {2, 1, 550, 70}, {2, 1, 550, 70},
                                       {2, 1, 550, 70}, {2, 1, 550, 70}, {2, 1, 550, 70}}}
    },
    // All dimensions are dynamic
    {
        {PartialShape{-1, -1, -1, -1}, {{2, 1, 32, 64}, {2, 2, 10, 20}, {2, 2, 100, 80},
                                        {2, 2, 10, 20}, {2, 1, 32, 64}, {2, 3, 64, 55}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 64, 128}, {2, 2, 20, 30}, {2, 2, 80, 120},
                                        {2, 2, 20, 30}, {1, 3, 64, 128}, {2, 3, 55, 128}}}
    },
    // Only K dimension is dynamic
    {
        {PartialShape{2, 2, 70, -1}, {{2, 2, 70, 512}, {2, 2, 70, 10}, {2, 2, 70, 33}, {2, 2, 70, 2000}, {2, 2, 70, 35}, {2, 2, 70, 600}}},
        {PartialShape{2, 2, -1, 70}, {{2, 2, 512, 70}, {2, 2, 10, 70}, {2, 2, 33, 70}, {2, 2, 2000, 70}, {2, 2, 35, 70}, {2, 2, 600, 70}}}
    },
    // Only N dimension is dynamic
    {
        {PartialShape{},              {{2, 2, 65, 550}}},
        {PartialShape{2, 2, 550, -1}, {{2, 2, 550, 70}, {2, 2, 550, 12}, {2, 2, 550, 70},
                                       {2, 2, 550, 12}, {2, 2, 550, 10}, {2, 2, 550, 64} }}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMul, MatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> transpose_b_shapes{
    { {{}, {{3, 3, 64, 64}}},   {{}, {{3, 3, 64, 64}}} },
    { {{}, {{1, 1, 32, 128}}},  {{}, {{1, 1, 64, 128}}} },
    { {{}, {{1, 1, 32, 128}}},  {{}, {{1, 1, 384, 128}}} },
    { {{}, {{1, 1, 64, 1500}}}, {{}, {{1, 1, 420, 1500}}} },
    { {{}, {{1, 1, 64, 1024}}}, {{}, {{1, 1, 420, 1024}}} },
    { {{}, {{4, 8, 32, 1024}}}, {{}, {{4, 8, 420, 1024}}} },
    // All dimensions are dynamic
    {
        {PartialShape{-1, -1, -1, -1}, {{2, 1, 32, 64},  {2, 2, 10, 20}, {2, 2, 100, 600}, {2, 1, 32, 64}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 3, 128, 64}, {2, 2, 30, 20}, {2, 2, 120, 600}, {1, 3, 128, 64}}}
    },
    // Only M is dynamic
    {
        {PartialShape{2, 2,  -1, 64}, {{2, 2, 40, 64},  {2, 2, 16, 64}}},
        {PartialShape{2, 2, 100, 64}, {{2, 2, 100, 64}, {2, 2, 100, 64}}}
    },
    // Only N is static
    {
        {PartialShape{2, 2, -1, 100}, {{2, 2, 32, 100},  {2, 2, 10, 100},  {2, 2, 10, 100}}},
        {PartialShape{2, 2, -1, 100}, {{2, 2, 100, 100}, {2, 2, 64, 100}, {2, 2, 100, 100}}}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulTransposeB, MatMulTransposeB,
                         ::testing::Combine(
                             ::testing::ValuesIn(transpose_b_shapes),
                             ::testing::ValuesIn(precisions()),
                             ::testing::Values(MatMulType::MatMul),
                             ::testing::Values(1), // MatMul
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> input_shapes_bias{
    { {{}, {{1, 2, 69, 43}}},   {{}, {{2, 1, 43, 49}}},    {{}, {{1, 1, 69, 49}}} },
    { {{}, {{1, 2, 95, 1023}}}, {{}, {{1, 2, 1023, 255}}}, {{}, {{1, 2, 95, 255}}} },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 2, 69, 43}, {1, 2, 95, 1023}, {1, 2, 69, 43}}},
        {PartialShape{-1, -1, -1, -1}, {{2, 1, 43, 49}, {1, 2, 1023, 255}, {2, 1, 43, 49}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 69, 49}, {1, 2, 95, 255}, {1, 1, 69, 49}}}
    },
    {
        {PartialShape{-1, -1, -1, -1}, {{2, 2, 16, 32}, {2, 2, 16, 32}, {2, 2, 16, 32}, {2, 2, 16, 32}}},
        {PartialShape{-1, -1, -1, -1}, {{2, 2, 32, 18}, {2, 2, 32, 18}, {2, 2, 32, 1},  {2, 2, 32, 1}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 16, 18}, {1, 1, 16, 1},  {1, 1, 16, 18}, {1, 1, 16, 1}}}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBias, MatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes_bias),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBiasQuantized, MatMulBiasQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes_bias),
                                 ::testing::ValuesIn(quantized_precisions_if_supported()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulsQuantized, MatMulsQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(SNIPPETS_TESTS_STATIC_SHAPES({{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}})),
                                 ::testing::ValuesIn(quantized_precisions_if_supported()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulsQuantizedSoftmax, MatMulsQuantizedSoftmax,
                         ::testing::Combine(
                                 ::testing::ValuesIn(SNIPPETS_TESTS_STATIC_SHAPES({{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}})),
                                 ::testing::ValuesIn(quantized_precisions_if_supported()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov