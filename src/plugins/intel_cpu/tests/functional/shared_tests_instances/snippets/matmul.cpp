// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_system_conf.h"

namespace ov {
namespace test {
namespace snippets {


namespace {
std::vector<std::vector<ov::PartialShape>> input_shapes{
        {{2, 1, 3, 5}, {1, 3, 5, 3}},
        {{3, 1, 32, 14}, {1, 2, 14, 32}},
        {{1, 2, 37, 23}, {2, 1, 23, 37}},
        {{1, 1, 37, 23}, {1, 2, 23, 33}},
        {{1, 1, 32, 23}, {1, 1, 23, 68}},
        {{1, 16, 384, 64}, {1, 16, 64, 384}},
        {{1, 1, 100, 700}, {1, 1, 700, 100}},
};

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
    if (!only_fp32) {
        auto quant = quantized_precisions();
        std::copy(quant.begin(), quant.end(), std::back_inserter(prc));
        // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
        if (ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx_bf16()) {
            prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
        }
    }
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
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 1, 43, 49}, {1, 1, 69, 49}}),
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBiasQuantized, MatMulBiasQuantized,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<std::vector<ov::PartialShape>>{
                                        std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 1, 43, 49}, {1, 2, 1, 1}},
                                        std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 1, 43, 49}, {1, 2, 69, 49}}}),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(1), // Subgraph
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulsQuantized, MatMulsQuantized,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}}),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulsQuantizedSoftmax, MatMulsQuantizedSoftmax,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 16, 128, 64}, {1, 16, 64, 128}, {128, 64}}),
                                 ::testing::ValuesIn(quantized_precisions()),
                                 ::testing::Values(3), // Subgraph + Reshape + Subgraph
                                 ::testing::Values(2), // Tokenized [MatMul+FQ+Matmul] and [FQ]
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov