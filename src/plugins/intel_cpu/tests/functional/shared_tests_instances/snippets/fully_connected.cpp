// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul.hpp"

#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {
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

std::vector<std::vector<ov::test::InputShape>> fc_input_shapes{
    {
        {PartialShape{-1, -1, -1, 2500}, {{2, 1, 32, 2500}, {1, 3, 80, 2500}}},
        {{}, {{2500, 256}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnected, MatMul,
                         ::testing::Combine(
                             ::testing::ValuesIn(fc_input_shapes),
                             ::testing::ValuesIn(precisions(false)),
                             ::testing::Values(MatMulType::FullyConnected),
                             ::testing::Values(1), // MatMul
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedFQ, MatMulFQ,
                         ::testing::Combine(
                                 ::testing::ValuesIn(fc_input_shapes),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(MatMulType::FullyConnected),
                                 ::testing::Values(1), // MatMul;
                                 ::testing::Values(1), // Tokenized MatMul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedEltwiseChain, MatMulEltwiseChain,
                         ::testing::Combine(
                             ::testing::ValuesIn(fc_input_shapes),
                             ::testing::ValuesIn(precisions()),
                             ::testing::Values(MatMulType::FullyConnected),
                             ::testing::Values(1), // MatMul
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> fc_cascade_shapes{
    {
        {PartialShape{-1, -1, -1, 2500}, {{2, 1, 32, 2500}, {1, 3, 80, 2500}, {2, 1, 32, 2500}}},
        {PartialShape{}, {{2500, 128}}},
        {PartialShape{}, {{128, 64}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedEltwiseChainCascade, MatMulEltwiseChainCascade,
                         ::testing::Combine(
                             ::testing::ValuesIn(fc_cascade_shapes),
                             ::testing::ValuesIn(precisions()),
                             ::testing::Values(MatMulType::FullyConnected),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> fc_transpose_b_shapes{
    {
        {PartialShape{-1, -1, -1, 2500}, {{2, 1, 32, 2500}, {1, 3, 80, 2500}}},
        {{}, {{256, 2500}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedTransposeB, MatMulTransposeB,
                         ::testing::Combine(
                             ::testing::ValuesIn(fc_transpose_b_shapes),
                             ::testing::ValuesIn(precisions(false)),
                             ::testing::Values(MatMulType::FullyConnected),
                             ::testing::Values(1), // MatMul
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);


std::vector<std::vector<ov::test::InputShape>> fc_bias_shapes{
    {
        {PartialShape{-1, -1, -1, 2500}, {{2, 1, 32, 2500}, {1, 3, 80, 2500}}},
        {{}, {{2500, 256}}},
        {PartialShape{-1, -1, -1, 256}, {{1, 1, 32, 256}, {1, 1, 80, 256}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedBias, MatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(fc_bias_shapes),
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(MatMulType::FullyConnected),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedBiasQuantized, MatMulBiasQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(fc_bias_shapes),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(MatMulType::FullyConnected),
                                 ::testing::Values(1), // Subgraph
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> fc_quantized_shapes{
    {
        {PartialShape{-1, -1, -1, 2500}, {{2, 1, 32, 2500}, {1, 3, 80, 2500}}},
        {{}, {{2500, 256}}},
        {{}, {{256, 64}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedsQuantized, MatMulsQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(fc_quantized_shapes),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(MatMulType::FullyConnected),
                                 ::testing::Values(1), // Reshape on weights is folded => only 1 Subgraph remains
                                 ::testing::Values(1), // Tokenized [MatMul+FQ+Matmul]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FullyConnectedsQuantizedSoftmax, MatMulsQuantizedSoftmax,
                         ::testing::Combine(
                                 ::testing::ValuesIn(fc_quantized_shapes),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(MatMulType::FullyConnected),
                                 ::testing::Values(1), // Reshape on weights is folded => only 1 Subgraph remains
                                 ::testing::Values(1), // Tokenized [MatMul+FQ+Matmul]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);
}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov