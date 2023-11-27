// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose_matmul.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_system_conf.h"

namespace ov {
namespace test {
namespace snippets {


namespace {
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
namespace transpose_zero_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{1, 49, 2, 23}, {2, 2, 23, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(0), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

// TODO: FuseTransposeToBrgemm supports fusing only if Transpose is before Parameter in cases when Transpose is on input at the moment
//       When we support the branch Parameter->FQ->Transpose->MatMul[0th input], uncomment this test case please
// INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ, TransposeMatMulFQ,
//                          ::testing::Combine(
//                                  ::testing::ValuesIn(transpose_input_shapes),
//                                  ::testing::Values(0), // Transpose on 0th Matmul input
//                                  ::testing::Values(ov::element::i8),
//                                  ::testing::Values(1), // MatMul
//                                  ::testing::Values(1), // Tokenized MatMul + FusedTranspose
//                                  ::testing::Values(ov::test::utils::DEVICE_CPU)),
//                          TransposeMatMulFQ::getTestCaseName);
} // namespace transpose_zero_input

namespace transpose_first_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 49, 13}, {1, 13, 3, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ, TransposeMatMulFQ,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMulFQ::getTestCaseName);
} // namespace transpose_first_input

namespace transpose_output {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 49, 13}, {1, 2, 13, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(2), // Transpose on Matmul output
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);

// TODO: At the moment we doesn't support the branch MatMul[output]->Transpose->FQ.
//      When we add support, uncomment this test case please
// INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ, TransposeMatMulFQ,
//                          ::testing::Combine(
//                                  ::testing::ValuesIn(transpose_input_shapes),
//                                  ::testing::Values(2), // Transpose on Matmul output
//                                  ::testing::Values(ov::element::i8),
//                                  ::testing::Values(1), // MatMul
//                                  ::testing::Values(1), // Tokenized MatMul + FusedTranspose
//                                  ::testing::Values(ov::test::utils::DEVICE_CPU)),
//                          TransposeMatMulFQ::getTestCaseName);
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
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}}),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExplicitTransposeMatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulBias, ExplicitTransposeMatMulBias,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}, {1, 1, 69, 49}}),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ExplicitTransposeMatMulBias::getTestCaseName);
} // namespace explicit_transpose

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov