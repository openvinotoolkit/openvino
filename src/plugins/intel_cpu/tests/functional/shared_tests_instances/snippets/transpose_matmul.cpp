// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose_matmul.hpp"

#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
static inline std::vector<std::vector<element::Type>> precisions(bool only_fp32 = true) {
    std::vector<std::vector<element::Type>> prc = precision_f32(2);
// Note: TPP doesn't support low precisions yet
#ifndef SNIPPETS_LIBXSMM_TPP
    if (!only_fp32) {
        auto quant = quantized_precisions_if_supported();
        std::copy(quant.begin(), quant.end(), std::back_inserter(prc));
        auto bfloat = precision_bf16_if_supported(2);
        std::copy(bfloat.begin(), bfloat.end(), std::back_inserter(prc));
    }
#endif
    return prc;
}
namespace transpose_zero_input {
const auto& transpose_input_shapes = SNIPPETS_TESTS_STATIC_SHAPES({{1, 49, 2, 23}, {2, 2, 23, 39}});
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(0), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> transpose_input_shapes_dynamic{
        {
                {PartialShape{-1, -1, -1, -1}, {{1, 49, 2, 23}, {1, 70, 2, 32}, {1, 49, 2, 23}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 2, 23, 39}, {2, 1, 32, 140}, {2, 2, 23, 39}}}
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynMatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes_dynamic),
                                 ::testing::Values(0), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> fc_transpose_input_shapes{
    {
        {PartialShape{-1, -1, -1, 2500}, {{1, 49, 2, 2500}, {1, 70, 2, 2500}, {1, 49, 2, 2500}}},
        {{}, {{2500, 256}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnected, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(fc_transpose_input_shapes),
                                 ::testing::Values(0), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(MatMulType::FullyConnected),
                                 ::testing::Values(1), // Fused MatMul + Transpose
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_zero_input

namespace transpose_first_input {
const auto& transpose_input_shapes = SNIPPETS_TESTS_STATIC_SHAPES({{2, 1, 49, 13}, {1, 13, 3, 39}});
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> transpose_input_shapes_dynamic{
        {
                {PartialShape{-1, -1, -1, -1}, {{2, 1, 49, 13}, {1, 2, 70, 30}, {2, 1, 49, 13}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 13, 3, 39}, {1, 30, 1, 80}, {1, 13, 3, 39}}}
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynMatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes_dynamic),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ, TransposeMatMulFQ,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMulFQ::getTestCaseName);
} // namespace transpose_first_input

namespace transpose_output {
const auto& transpose_input_shapes = SNIPPETS_TESTS_STATIC_SHAPES({{2, 1, 49, 13}, {1, 2, 13, 39}});

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(2), // Transpose on Matmul output
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> transpose_input_shapes_dynamic{
        {
                {PartialShape{-1, -1, -1, -1}, {{2, 1, 49, 13}, {1, 2, 70, 30}, {2, 1, 49, 13}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 2, 13, 49}, {1, 1, 30, 70}, {1, 2, 13, 49}}}
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynMatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes_dynamic),
                                 ::testing::Values(2), // Transpose on Matmul output
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> fc_transpose_input_shapes{
    {
        {PartialShape{-1, -1, -1, 2500}, {{2, 1, 49, 2500}, {1, 2, 70, 2500}, {2, 1, 49, 2500}}},
        {{}, {{2500, 256}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnected, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(fc_transpose_input_shapes),
                                 ::testing::Values(2), // Transpose on Matmul output
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(MatMulType::FullyConnected),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_output

namespace explicit_transpose {
static inline std::vector<std::vector<element::Type>> precisions(bool only_fp32 = true) {
    std::vector<std::vector<element::Type>> prc = {
            {element::f32, element::f32},
    };
    if (!only_fp32) {
        // In Snippets MatMul INT8 is supported only on VNNI/AMX platforms
        if (ov::with_cpu_x86_avx512_core_vnni() || ov::with_cpu_x86_avx512_core_amx_int8()) {
            prc.emplace_back(std::vector<element::Type>{element::i8, element::i8});
            prc.emplace_back(std::vector<element::Type>{element::u8, element::i8});
        }
        // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
        if (ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx_bf16()) {
            prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
        }
    }
    return prc;
}
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExplicitTransposeMatMul, ExplicitTransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(SNIPPETS_TESTS_STATIC_SHAPES({{1, 2, 69, 43}, {2, 49, 2, 43}})),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExplicitTransposeMatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> explicit_input_shapes_dynamic{
        {
                {PartialShape{-1, -1, -1, -1}, {{1, 2, 69, 43}, {1, 1, 70, 40}, {1, 2, 69, 43}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 49, 2, 43}, {2, 70, 1, 40}, {2, 49, 2, 43}}}
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynExplicitTransposeMatMul, ExplicitTransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(explicit_input_shapes_dynamic),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExplicitTransposeMatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulBias, ExplicitTransposeMatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(SNIPPETS_TESTS_STATIC_SHAPES({{1, 2, 69, 43}, {2, 49, 2, 43}, {1, 1, 69, 49}})),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExplicitTransposeMatMulBias::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> explicit_bias_input_shapes_dynamic{
        {
                {PartialShape{-1, -1, -1, -1}, {{1, 2, 69, 43}, {1, 1, 70, 40}, {1, 2, 69, 43}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 49, 2, 43}, {2, 60, 1, 40}, {2, 49, 2, 43}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 2, 69, 49}, {2, 1, 70, 60}, {2, 2, 69, 49}}},
        },
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynTransposeMatMulBias, ExplicitTransposeMatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(explicit_bias_input_shapes_dynamic),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions(true)),
                                 ::testing::Values(MatMulType::MatMul),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExplicitTransposeMatMulBias::getTestCaseName);
} // namespace explicit_transpose

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov