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
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes_5D {{{10, 3, 2, 49, 23}, {10, 3, 2, 23, 39}}};
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes_4D {{{2, 7, 49, 23}, {2, 7, 23, 39}}};
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes_3D {{{2, 49, 23}, {2, 23, 39}}};

std::vector<int32_t> order_03124 {0, 3, 1, 2, 4};
std::vector<int32_t> order_03142 {0, 3, 1, 4, 2};
std::vector<int32_t> order_0213 {0, 2, 1, 3};
std::vector<int32_t> order_2013 {2, 0, 1, 3};
std::vector<int32_t> order_0231 {0, 2, 3, 1};
std::vector<int32_t> order_2031 {2, 0, 3, 1};
std::vector<int32_t> order_102 {1, 0, 2};
std::vector<int32_t> order_120 {1, 2, 0};

std::vector<std::vector<ov::PartialShape>> order_shapes(std::vector<std::vector<ov::PartialShape>> shapes, const std::vector<int>& order, size_t idx) {
    auto order_shape = [&](ov::PartialShape& shape) {
        ov::PartialShape new_shape(shape);
        for (size_t i = 0; i < order.size(); i++) {
            new_shape[order[i]] = shape[i];
        }
        shape = new_shape;
    };
    for (auto& inputs : shapes) {
        order_shape(inputs[idx]);
    }
    return shapes;
}

static inline std::vector<std::vector<element::Type>> precisions(bool only_fp32 = true) {
    std::vector<std::vector<element::Type>> prc = {
            {element::f32, element::f32},
    };
    if (!only_fp32) {
        // In Snippets MatMul INT8 is supported only on VNNI/AMX platforms
        if (InferenceEngine::with_cpu_x86_avx512_core_vnni() || InferenceEngine::with_cpu_x86_avx512_core_amx_int8()) {
            prc.emplace_back(std::vector<element::Type>{element::i8, element::i8});
            prc.emplace_back(std::vector<element::Type>{element::u8, element::i8});
        }
        // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
        if (InferenceEngine::with_cpu_x86_bfloat16() || InferenceEngine::with_cpu_x86_avx512_core_amx_bf16()) {
            prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
        }
    }
    return prc;
}
namespace transpose_zero_input {
size_t idx = 0;
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_5D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_5D, order_03124, idx)),
                                 ::testing::Values(order_03124),
                                 ::testing::Values(idx), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_4D_0213, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_4D, order_0213, idx)),
                                 ::testing::Values(order_0213),
                                 ::testing::Values(idx), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_4D_2013, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_4D, order_2013, idx)),
                                 ::testing::Values(order_2013),
                                 ::testing::Values(idx), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_3D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_3D, order_102, idx)),
                                 ::testing::Values(order_102),
                                 ::testing::Values(idx), // Transpose on 0th Matmul input
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
size_t idx = 1;
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_5D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_5D, order_03124, idx)),
                                 ::testing::Values(order_03124),
                                 ::testing::Values(idx), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_4D_0213, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_4D, order_0213, idx)),
                                 ::testing::Values(order_0213),
                                 ::testing::Values(idx), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_4D_2013, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_4D, order_2013, idx)),
                                 ::testing::Values(order_2013),
                                 ::testing::Values(idx), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_3D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_3D, order_102, idx)),
                                 ::testing::Values(order_102),
                                 ::testing::Values(idx), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions(false)),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ_5D, TransposeMatMulFQ,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_5D, order_03142, idx)),
                                 ::testing::Values(order_03142),
                                 ::testing::Values(idx), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMulFQ::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulFQ_4D_0213, TransposeMatMulFQ,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_4D, order_0231, idx)),
                                 ::testing::Values(order_0231),
                                 ::testing::Values(idx), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMulFQ::getTestCaseName);


} // namespace transpose_first_input

namespace transpose_output {
size_t idx = 2;
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_5D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes_5D),
                                 ::testing::Values(order_03124),
                                 ::testing::Values(idx),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_4D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes_4D),
                                 ::testing::ValuesIn({order_0213, order_2013}),
                                 ::testing::Values(idx),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // MatMul
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult_3D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes_3D),
                                 ::testing::Values(order_102),
                                 ::testing::Values(idx),
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
        if (InferenceEngine::with_cpu_x86_avx512_core_vnni() || InferenceEngine::with_cpu_x86_avx512_core_amx_int8()) {
            prc.emplace_back(std::vector<element::Type>{element::i8, element::i8});
            prc.emplace_back(std::vector<element::Type>{element::u8, element::i8});
        }
        // In Snippets MatMul BF16 is supported only on bf16/AMX platforms
        if (InferenceEngine::with_cpu_x86_bfloat16() || InferenceEngine::with_cpu_x86_avx512_core_amx_bf16()) {
            prc.emplace_back(std::vector<element::Type>{element::bf16, element::bf16});
        }
    }
    return prc;
}
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExplicitTransposeMatMul_5D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_5D, order_03142, 1)),
                                 ::testing::Values(order_03142),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExplicitTransposeMatMul_3D, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(order_shapes(transpose_input_shapes_3D, order_120, 1)),
                                 ::testing::Values(order_120),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExplicitTransposeMatMulBias, TransposeMatMulBias,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}, {1, 1, 69, 49}}),
                                 ::testing::Values(std::vector<int>{0, 2, 3, 1}),
                                 ::testing::Values(1), // Transpose on second input
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMatMulBias::getTestCaseName);
} // namespace explicit_transpose

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov