// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {
std::vector<ov::PartialShape> input_shapes_4D{{2, 3, 5, 13}, {2, 3, 2, 4}, {1, 7, 1, 4}};
std::vector<ov::PartialShape> input_shapes_3D{{3, 5, 13}, {3, 2, 4}, {7, 1, 4}};

std::vector<std::vector<int32_t>> orders_4D{{0, 2, 3, 1}};
std::vector<std::vector<int32_t>> orders_3D{{1, 2, 0}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Transpose_3D, Transpose,
                     ::testing::Combine(
                             ::testing::ValuesIn(input_shapes_3D),
                             ::testing::ValuesIn(orders_3D),
                             ::testing::Values(1), // Transpose
                             ::testing::Values(1), // Tokenized Transpose
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     Transpose::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Transpose_4D, Transpose,
                     ::testing::Combine(
                             ::testing::ValuesIn(input_shapes_4D),
                             ::testing::ValuesIn(orders_4D),
                             ::testing::Values(1), // Transpose
                             ::testing::Values(1), // Tokenized Transpose
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     Transpose::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMul, TransposeMul,
                         ::testing::Combine(
                                 ::testing::Values(ov::PartialShape {2, 31, 3, 5}),
                                 ::testing::ValuesIn(std::vector<ov::PartialShape>{{2, 3, 5, 31}}),
                                 ::testing::Values(std::vector<int> {0, 2, 3, 1}),
                                 ::testing::Values(1), // Transpose
                                 ::testing::Values(1), // Tokenized Transpose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMul::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov