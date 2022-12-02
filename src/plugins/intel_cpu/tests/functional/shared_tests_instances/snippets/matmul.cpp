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
        {{2, 1, 3, 5}, {1, 3, 5, 3}},
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
                             ::testing::Values(3), // Sinh * 2 + MatMu;
                             ::testing::Values(1), // Tokenized MatMul
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MatMul::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov