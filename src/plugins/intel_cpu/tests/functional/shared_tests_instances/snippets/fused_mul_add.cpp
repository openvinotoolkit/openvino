// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/fused_mul_add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

const std::vector<ov::PartialShape> in_shapes_2 = {{1, 64, 10,  1}, {}};
const std::vector<size_t> input_idxes = {0, 1};
const std::vector<bool> constant_input = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_FusedMulAdd, FusedMulAdd,
                        ::testing::Combine(
                             ::testing::Values(ov::PartialShape{1, 64, 10, 10}),
                             ::testing::Values(ov::PartialShape{1, 64, 10,  1}),
                             ::testing::ValuesIn(in_shapes_2),
                             ::testing::ValuesIn(input_idxes),
                             ::testing::ValuesIn(constant_input),
                             ::testing::Values(1), // Subgraph with FMA
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         FusedMulAdd::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov