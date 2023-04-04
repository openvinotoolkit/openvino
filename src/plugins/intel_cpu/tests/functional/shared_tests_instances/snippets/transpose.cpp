// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {
std::vector<ov::PartialShape> input_shapes{{2, 3, 5, 13}, {2, 3, 2, 4}, {1, 7, 1, 4}};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Transpose, Transpose,
                     ::testing::Combine(
                             ::testing::ValuesIn(input_shapes),
                             ::testing::Values(std::vector<int> {0, 2,  3, 1}),
                             ::testing::Values(1), // Transpose
                             ::testing::Values(1), // Tokenized Transpose
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     Transpose::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov