// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {
std::vector<std::vector<ov::PartialShape>> input_shapes{
        {{1, 1, 3, 5}, {1, 3, 5, 3}},
        {{4, 1, 32, 64}, {1, 2, 64, 32}},
        {{2, 1, 32, 64}, {1, 2, 64, 17}},
        {{4, 1, 5, 17}, {1, 3, 17, 5}},
        {{1, 2, 27, 43}, {2, 1, 43, 27}},
        {{1, 2, 27, 43}, {2, 1, 43, 33}},
        {{1, 2, 37, 43}, {2, 1, 43, 37}},
        {{1, 2, 37, 43}, {2, 1, 43, 33}},
        {{1, 2, 69, 43}, {2, 1, 43, 49}}
};
std::vector<element::Type> precisions{element::f32};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMul, MatMul,
                     ::testing::Combine(
                             ::testing::ValuesIn(input_shapes),
                             ::testing::ValuesIn(precisions),
                             ::testing::Values(3), // Sinh * 2 + MatMu;
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBias, MatMulBias,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 1, 43, 49}, {1, 1, 69, 49}}),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(4), // Sinh * 3 + Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExplicitTransposeMatMul, ExplicitTransposeMatMul,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}}),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(3), // Sinh * 2 + Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ExplicitTransposeMatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulBias, ExplicitTransposeMatMulBias,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}, {1, 1, 69, 49}}),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(4), // Sinh * 3 + Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMulMatMulBias, ExplicitTransposeMulMatMulBias,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}, {1, 2, 1, 1}, {1, 1, 69, 49}}),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(5), // Sinh * 4 + Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

namespace transpose_zero_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 69, 3, 43}, {2, 3, 43, 49}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(0), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(3), // Sinh * 2 + MatMu;
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_zero_input

namespace transpose_first_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 69, 43}, {1, 43, 3, 49}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(3), // Sinh * 2 + MatMu;
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_first_input

namespace transpose_output {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 69, 43}, {1, 3, 43, 49}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(2), // Transpose on Matmul output
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(3), // Sinh * 2 + MatMu;
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_output

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov