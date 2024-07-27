// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul.hpp"

#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov {
namespace test {
namespace snippets {

#define STATIC_SHAPES(...) static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{__VA_ARGS__})

namespace {
const auto& input_shapes = STATIC_SHAPES(
        {{2, 1, 3, 5}, {1, 3, 5, 3}},
        {{3, 1, 32, 14}, {1, 2, 14, 32}},
        {{1, 2, 37, 23}, {2, 1, 23, 37}},
        {{1, 1, 37, 23}, {1, 2, 23, 33}},
        {{1, 1, 32, 23}, {1, 1, 23, 68}},
        {{1, 16, 384, 64}, {1, 16, 64, 384}},
        {{1, 1, 100, 700}, {1, 1, 700, 100}},
        {{1, 1, 100, 1024}, {1, 1, 1024, 100}},
        {{1, 1, 100, 2500}, {1, 1, 2500, 100}},
        {{1, 1, 100, 4500}, {1, 1, 4500, 100}},
);

static inline std::vector<std::vector<element::Type>> quantized_precisions() {
    std::vector<std::vector<element::Type>> prc = {};
    // In Snippets MatMul INT8 is supported only on VNNI/AMX platforms
    if (ov::with_cpu_x86_avx512_core_vnni() || ov::with_cpu_x86_avx512_core_amx_int8()) {
        prc.emplace_back(std::vector<element::Type>{element::i8, element::i8});
        prc.emplace_back(std::vector<element::Type>{element::u8, element::i8});
    }
    return prc;
}

static inline std::vector<std::vector<element::Type>> precisions(bool only_fp32 = true) {
    std::vector<std::vector<element::Type>> prc = {
            {element::f32, element::f32},
    };
// Note: TPP doesn't support low precisions yet
#ifndef SNIPPETS_LIBXSMM_TPP
    if (!only_fp32) {
        auto quant = quantized_precisions();
        std::copy(quant.begin(), quant.end(), std::back_inserter(prc));
        // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
        if (ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx_bf16()) {
            prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
        }
    }
#endif
    return prc;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, MatMul,
                         ::testing::Combine(
                             ::testing::ValuesIn(input_shapes),
                             ::testing::ValuesIn(precisions(false)),
                             ::testing::Values(1), // MatMul
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);


std::vector<std::vector<ov::test::InputShape>> input_shapes_dynamic{
        // All dimensions are dynamic
        {
            {PartialShape{-1, -1, -1, -1}, {{2, 1, 32, 64}, {2, 2, 10, 20}, {2, 2, 100, 80},
                                            {2, 2, 10, 20}, {2, 1, 32, 64}, {2, 3, 64, 55}}},
            {PartialShape{-1, -1, -1, -1}, {{1, 3, 64, 128}, {2, 2, 20, 30}, {2, 2, 80, 120},
                                            {2, 2, 20, 30}, {1, 3, 64, 128}, {2, 3, 55, 128}}}
        },
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
        // Only K dimension is dynamic
        {
            {PartialShape{2, 2, 70, -1}, {{2, 2, 70, 128}, {2, 2, 70, 10}, {2, 2, 70, 33},
                                          {2, 2, 70, 35},  {2, 2, 70, 100}}},
            {PartialShape{2, 2, -1, 70}, {{2, 2, 128, 70}, {2, 2, 10, 70}, {2, 2, 33, 70},
                                          {2, 2, 35, 70}, {2, 2, 100, 70}}}
        },
        // Only N dimension is dynamic
        {
            {PartialShape{},              {{2, 2, 65, 550}}},
            {PartialShape{2, 2, 550, -1}, {{2, 2, 550, 70}, {2, 2, 550, 12}, {2, 2, 550, 70},
                                           {2, 2, 550, 12}, {2, 2, 550, 10}}}
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynMatMul, MatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes_dynamic),
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulFQ, MatMulFQ,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul;
                                 ::testing::Values(1), // Tokenized MatMul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBias, MatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(STATIC_SHAPES({{1, 2, 69, 43}, {2, 1, 43, 49}, {1, 1, 69, 49}},
                                                                   {{1, 2, 95, 1023}, {1, 2, 1023, 255}, {1, 2, 95, 255}})),
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> input_shapes_dynamic_bias{
        {
                {PartialShape{-1, -1, -1, -1}, {{1, 2, 69, 43}, {1, 2, 95, 1023}, {1, 2, 69, 43}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 1, 43, 49}, {1, 2, 1023, 255}, {2, 1, 43, 49}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 1, 69, 49}, {1, 2, 95, 255}, {1, 1, 69, 49}}}
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynMatMulBias, MatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes_dynamic_bias),
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBiasQuantized, MatMulBiasQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(STATIC_SHAPES({{1, 2, 69, 43}, {2, 1, 43, 49}, {1, 2, 1, 1}},
                                                                    {{1, 2, 69, 43}, {2, 1, 43, 49}, {1, 2, 69, 49}})),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(1), // Subgraph
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulQuantized, MatMulQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(STATIC_SHAPES({{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}})),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulQuantizedSoftmax, MatMulQuantizedSoftmax,
                         ::testing::Combine(
                                 ::testing::ValuesIn(STATIC_SHAPES({{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}})),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov