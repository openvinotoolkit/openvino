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

static inline std::vector<std::vector<element::Type>> quantized_precisions() {
    std::vector<std::vector<element::Type>> prc = {};
// Note: TPP doesn't support low precisions yet
#ifndef SNIPPETS_LIBXSMM_TPP
    // In Snippets MatMul INT8 is supported only on VNNI/AMX platforms
    if (ov::with_cpu_x86_avx512_core_vnni() || ov::with_cpu_x86_avx512_core_amx_int8()) {
        prc.emplace_back(std::vector<element::Type>{element::i8, element::i8});
        prc.emplace_back(std::vector<element::Type>{element::u8, element::i8});
    }
#endif
    return prc;
}

static inline std::vector<std::vector<element::Type>> half_precisions() {
    std::vector<std::vector<element::Type>> prc = {};
// Note: TPP doesn't support low precisions yet
#ifndef SNIPPETS_LIBXSMM_TPP
    // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
    if (ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx_bf16()) {
        prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
    }
#endif
    return prc;
}

static inline std::vector<std::vector<element::Type>> low_precisions() {
    std::vector<std::vector<element::Type>> prc = quantized_precisions();
    auto half = half_precisions();
    std::copy(half.begin(), half.end(), std::back_inserter(prc));
    return prc;
}

std::vector<std::vector<ov::test::InputShape>> get_input_shapes(bool is_fp32 = true) {
    std::vector<std::vector<ov::test::InputShape>> shapes = {
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
    };
    // [150842] Need to support dynamic K, N on AMX platforms
    if (is_fp32 || !with_cpu_x86_avx512_core_amx()) {
        // All dimensions are dynamic
        shapes.push_back({
                {PartialShape{-1, -1, -1, -1}, {{2, 1, 32, 64}, {2, 2, 10, 20}, {2, 2, 100, 80},
                                                {2, 2, 10, 20}, {2, 1, 32, 64}, {2, 3, 64, 55}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 3, 64, 128}, {2, 2, 20, 30}, {2, 2, 80, 120},
                                                {2, 2, 20, 30}, {1, 3, 64, 128}, {2, 3, 55, 128}}}
        });
        // Only K dimension is dynamic
        shapes.push_back({
            {PartialShape{2, 2, 70, -1}, {{2, 2, 70, 512}, {2, 2, 70, 10}, {2, 2, 70, 33}, {2, 2, 70, 2000}, {2, 2, 70, 35}, {2, 2, 70, 600}}},
            {PartialShape{2, 2, -1, 70}, {{2, 2, 512, 70}, {2, 2, 10, 70}, {2, 2, 33, 70}, {2, 2, 2000, 70}, {2, 2, 35, 70}, {2, 2, 600, 70}}}
        });
        // Only N dimension is dynamic
        shapes.push_back({
                {PartialShape{},              {{2, 2, 65, 550}}},
                {PartialShape{2, 2, 550, -1}, {{2, 2, 550, 70}, {2, 2, 550, 12}, {2, 2, 550, 70},
                                               {2, 2, 550, 12}, {2, 2, 550, 10}, {2, 2, 550, 64} }}
        });
    }
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMul_FP32, MatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(get_input_shapes()),
                                 ::testing::Values(std::vector<element::Type>{element::f32, element::f32}),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMul_LowPrc, MatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(get_input_shapes(false)),
                                 ::testing::ValuesIn(low_precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> get_transpose_b_shapes(bool is_fp32 = true) {
    std::vector<std::vector<ov::test::InputShape>> shapes = {
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
    };
    // [150842] Need to support dynamic K, N on AMX platforms
    if (is_fp32 || !with_cpu_x86_avx512_core_amx()) {
        // Only N is static
        shapes.push_back({
                {PartialShape{2, 2, -1, -1},  {{2, 2, 32, 64},  {2, 2, 10, 20},  {2, 2, 10, 20}}},
                {PartialShape{2, 2, 100, -1}, {{2, 2, 100, 64}, {2, 2, 100, 20}, {2, 2, 100, 20}}}
        });
    }
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulTransposeB_FP32, MatMulTransposeB,
                         ::testing::Combine(
                             ::testing::ValuesIn(get_transpose_b_shapes(true)),
                             ::testing::Values(std::vector<element::Type>{element::f32, element::f32}),
                             ::testing::Values(MatMulType::MatMul),
                             ::testing::Values(1), // MatMul
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulTransposeB_LowPrc, MatMulTransposeB,
                         ::testing::Combine(
                             ::testing::ValuesIn(get_transpose_b_shapes(false)),
                             ::testing::ValuesIn(low_precisions()),
                             ::testing::Values(MatMulType::MatMul),
                             ::testing::Values(1), // MatMul
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> get_input_shapes_bias(bool is_fp32 = true) {
    std::vector<std::vector<ov::test::InputShape>> shapes = {
        { {{}, {{1, 2, 69, 43}}},   {{}, {{2, 1, 43, 49}}},    {{}, {{1, 1, 69, 49}}} },
        { {{}, {{1, 2, 95, 1023}}}, {{}, {{1, 2, 1023, 255}}}, {{}, {{1, 2, 95, 255}}} },
    };
    // [150842] Need to support dynamic K, N on AMX platforms
    if (is_fp32 || !with_cpu_x86_avx512_core_amx()) {
        shapes.push_back({
                {PartialShape{-1, -1, -1, -1}, {{1, 2, 69, 43}, {1, 2, 95, 1023}, {1, 2, 69, 43}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 1, 43, 49}, {1, 2, 1023, 255}, {2, 1, 43, 49}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 1, 69, 49}, {1, 2, 95, 255}, {1, 1, 69, 49}}}
        });
        shapes.push_back({
                {PartialShape{-1, -1, -1, -1}, {{2, 2, 16, 32}, {2, 2, 16, 32}, {2, 2, 16, 32}, {2, 2, 16, 32}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 2, 32, 18}, {2, 2, 32, 18}, {2, 2, 32, 1},  {2, 2, 32, 1}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 1, 16, 18}, {1, 1, 16, 1},  {1, 1, 16, 18}, {1, 1, 16, 1}}}
        });
    }
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBias_FP32, MatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(get_input_shapes_bias()),
                                 ::testing::Values(std::vector<element::Type>{element::f32, element::f32}),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBias_LowPrc, MatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(get_input_shapes_bias(false)),
                                 ::testing::ValuesIn(low_precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBiasQuantized, MatMulBiasQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(get_input_shapes_bias(false)),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulsQuantized, MatMulsQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(STATIC_SHAPES({{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}})),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulsQuantizedSoftmax, MatMulsQuantizedSoftmax,
                         ::testing::Combine(
                                 ::testing::ValuesIn(STATIC_SHAPES({{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}})),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov