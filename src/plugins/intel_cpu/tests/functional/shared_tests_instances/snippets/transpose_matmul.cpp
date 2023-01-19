// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose_matmul.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {
std::vector<element::Type> precisions{element::f32};
namespace transpose_zero_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{1, 49, 2, 23}, {2, 2, 23, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(0), // Transpose on 0th Matmul input
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(1), // MatMul;
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_zero_input

namespace transpose_first_input {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 49, 13}, {1, 13, 3, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(1), // Transpose on 1st Matmul input
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(1), // MatMu;
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_first_input

namespace transpose_output {
std::vector<std::vector<ov::PartialShape>> transpose_input_shapes{
        {{2, 1, 49, 13}, {1, 2, 13, 39}}
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, TransposeMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(transpose_input_shapes),
                                 ::testing::Values(2), // Transpose on Matmul output
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(1), // MatMu;
                                 ::testing::Values(1), // Tokenized MatMul + FusedTranspose
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         TransposeMatMul::getTestCaseName);
} // namespace transpose_output

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov