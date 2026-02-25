// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<InputShape> input_shapes_3D = {
    {{}, {{3, 5, 13}}},
    {{}, {{3, 2, 4}}},
    {{}, {{7, 1, 4}}},
    {{-1, -1, -1}, {{7, 1, 4}, {5, 7, 18}, {5, 7, 18}, {7, 1, 4}}},
    {{10, -1, -1}, {{10, 1, 4}, {10, 17, 18}, {10, 7, 18}, {10, 1, 4}}},
    {{10, -1, 15}, {{10, 1, 15}, {10, 1, 15}, {10, 7, 15}, {10, 15, 15}}},
};

const std::vector<InputShape> input_shapes_4D = {
    {{}, {{2, 3, 5, 13}}},
    {{}, {{2, 3, 2, 4}}},
    {{}, {{1, 7, 1, 4}}},
    {{-1, -1, -1, -1}, {{1, 7, 1, 4}, {2, 3, 2, 4}, {8, 7, 1, 4}, {2, 3, 2, 4}, {1, 7, 1, 4}}},
    {{-1, 9, -1, -1}, {{1, 9, 1, 4}, {2, 9, 2, 4}, {8, 9, 1, 4}, {1, 9, 1, 4}, {1, 9, 1, 4}, {2, 9, 2, 4}}},
    {{-1, -1, -1, 5}, {{2, 8, 2, 5}, {2, 8, 2, 5}, {8, 2, 5, 5}, {1, 2, 5, 5}, {8, 3, 5, 5}}},
    {{-1, 9, -1, 5}, {{1, 9, 1, 5}, {2, 9, 2, 5}, {8, 9, 5, 5}, {1, 9, 5, 5}, {8, 9, 5, 5}}},
    {{2, -1, -1, -1}, {{2, 7, 1, 4}, {2, 3, 2, 4}, {2, 7, 1, 4}, {2, 3, 2, 4}, {2, 7, 1, 4}}},
};

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

const std::vector<std::pair<InputShape, InputShape>> inputShapesPair = {
    {{{}, {{2, 31, 3, 5}}}, {{}, {{2, 3, 5, 31}}}},
    {{{-1, -1, -1, -1}, {{2, 31, 3, 5}, {2, 33, 4, 5}, {2, 33, 4, 5}}},
     {{-1, -1, -1, -1}, {{2, 3, 5, 31}, {2, 4, 5, 1}, {2, 4, 5, 1}}}},
    {{{-1, -1, -1, -1}, {{2, 31, 3, 5}, {2, 33, 4, 5}, {2, 33, 4, 5}}},
     {{-1, -1, 1, -1}, {{2, 3, 1, 31}, {2, 4, 1, 33}, {2, 4, 1, 1}}}},
    {{{-1, 33, -1, -1}, {{2, 33, 3, 5}, {2, 33, 4, 5}, {2, 33, 2, 5}}},
     {{-1, -1, 1, 1}, {{2, 3, 1, 1}, {2, 4, 1, 1}, {2, 2, 1, 1}}}},
    {{{-1, -1, -1, -1}, {{2, 16, 3, 5}, {2, 8, 4, 5}, {2, 4, 2, 5}}},
     {{-1, -1, -1, 1}, {{2, 3, 1, 1}, {2, 4, 1, 1}, {2, 2, 1, 1}}}},
    {{{-1, 18, -1, -1}, {{2, 18, 3, 5}, {2, 18, 4, 5}, {2, 18, 2, 6}}},
     {{-1, -1, -1, 18}, {{2, 3, 5, 18}, {2, 4, 5, 18}, {2, 2, 6, 18}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeMul, TransposeMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesPair),
                                 ::testing::Values(std::vector<int> {0, 2, 3, 1}),
                                 ::testing::Values(1), // Transpose
                                 ::testing::Values(1), // Tokenized Transpose
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeMul::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov