// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<ov::Shape> inputShape = {
        {1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHA,
                     ::testing::Combine(
                             ::testing::Values(inputShape),
                             ::testing::Values(5),  // Subgraph + 4xSin
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     MHA::getTestCaseName);

const std::vector<std::vector<ov::Shape>> inputShapeSelect = {
        {  // without broadcast
            {1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 128, 12, 64}
        },
        {  // with broadcast
            {1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 1, 1}, {1, 12, 1, 1}, {1, 128, 12, 64}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHASelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeSelect),
                                 ::testing::Values(8),  // Subgraph(MHA) + 6xSin + Subgraph(Less)
                                 ::testing::Values(2),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);


} // namespace
} // namespace snippets
} // namespace test
} // namespace ov