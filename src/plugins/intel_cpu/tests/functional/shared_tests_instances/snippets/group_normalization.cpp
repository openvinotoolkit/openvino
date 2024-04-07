// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/group_normalization.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

// this config make sure group_num(1024) is bigger than thread_number,
// and subtensor(4*8*2(2048/1024)*4(sizeof(float)) = 256 Byte) is smaller than l1 per core cache,
// which to tokenize GN as subgraph.
const std::vector<ov::Shape> inputShape = {
    {{1, 2048, 4, 8}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GroupNormalization, GroupNormalization,
                     ::testing::Combine(
                             ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)),
                             ::testing::Values(1024),    // num_group
                             ::testing::Values(0.001),   // eps
                             ::testing::Values(1),       // expected node number
                             ::testing::Values(1),       // expected subgraph number
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     GroupNormalization::getTestCaseName);

} // namespace

} // namespace snippets
} // namespace test
} // namespace ov