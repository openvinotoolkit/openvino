// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/group_normalization.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

// snippets ignore_callback is set in setup, so these tests will always run as snippets
const std::vector<ov::Shape> inputShape = {
    {3, 8},
    {3, 8, 1},
    {3, 8, 7},
    {3, 8, 16},
    {3, 8, 21},
    {1, 4, 8, 8},
    {1, 8, 1, 22},
    {1, 16, 1, 33},
    {1, 4, 1, 1, 34},
    {1, 8, 1, 8, 2, 2},
    {1, 8, 1, 8, 2, 2, 2}
};

const std::vector<size_t> numGroups = {
    2, 4,
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GroupNormalization, GroupNormalization,
                     ::testing::Combine(
                             ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)),
                             ::testing::ValuesIn(numGroups),    // num_group
                             ::testing::Values(0.0001),         // eps
                             ::testing::Values(1),              // expected node number
                             ::testing::Values(1),              // expected subgraph number
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     GroupNormalization::getTestCaseName);

} // namespace

} // namespace snippets
} // namespace test
} // namespace ov