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
        {{1, 1, 7, 5}, {1, 1, 5, 3}},
        {{3, 1, 32, 14}, {1, 2, 14, 32}},
        {{1, 2, 37, 23}, {2, 1, 23, 37}},
        {{1, 1, 37, 23}, {1, 2, 23, 33}},
        {{2, 1, 69, 43}, {1, 1, 43, 49}}
};
std::vector<element::Type> precisions{element::f32};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMult, MatMul,
                         ::testing::Combine(
                             ::testing::ValuesIn(input_shapes),
                             ::testing::ValuesIn(precisions),
                             ::testing::Values(1), // MatMu;
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

std::vector<std::vector<ov::PartialShape>> input_shapes_bias{
        {{1, 1, 128, 43}, {1, 1, 43, 64}, {1, 1, 128, 64}},
        {{1, 1, 64, 43}, {1, 1, 43, 32}, {1, 1, 64, 32}},
        {{2, 1, 64, 43}, {1, 1, 43, 32}, {1, 3, 64, 32}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MatMulBias, MatMulBias,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes_bias),
                                 ::testing::ValuesIn(precisions),
                                 ::testing::Values(1), // Subgraph;
                                 ::testing::Values(1), // Tokenized MatMul+Bias
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExplicitTransposeMatMul, ExplicitTransposeMatMul,
//                         ::testing::Combine(
//                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}}),
//                                 ::testing::ValuesIn(precisions),
//                                 ::testing::Values(1), // Subgraph;
//                                 ::testing::Values(1), // Tokenized MatMul+Bias
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                         ExplicitTransposeMatMul::getTestCaseName);

//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMatMulBias, ExplicitTransposeMatMulBias,
//                         ::testing::Combine(
//                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}, {1, 1, 69, 49}}),
//                                 ::testing::ValuesIn(precisions),
//                                 ::testing::Values(1), // Subgraph;
//                                 ::testing::Values(1), // Tokenized MatMul+Bias
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                         MatMul::getTestCaseName);
//
//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMulMatMulBias, ExplicitTransposeMulMatMulBias,
//                         ::testing::Combine(
//                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 2, 69, 43}, {2, 49, 2, 43}, {1, 2, 1, 1}, {1, 1, 69, 49}}),
//                                 ::testing::ValuesIn(precisions),
//                                 ::testing::Values(1), // Subgraph;
//                                 ::testing::Values(1), // Tokenized MatMul+Bias
//                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                         MatMul::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov