// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/group_normalization.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<ov::Shape> inputShape = {
    {{1, 512, 20, 1024}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GroupNormalization, GroupNormalization,
                     ::testing::Combine(
                             ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)),
                             ::testing::Values(256),     // num_group
                             ::testing::Values(0.001),   // eps
                             ::testing::Values(6),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     GroupNormalization::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov