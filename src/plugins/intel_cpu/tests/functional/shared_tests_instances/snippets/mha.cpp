// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 1, 1, 128}, {1, 128, 16, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 16, 1, 1}, {1, 128, 16, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHA,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShapes),
                             ::testing::ValuesIn({false, true}),
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
                                 ::testing::Values(false),  // Need to support True for graph builder in tests
                                 ::testing::Values(8),  // Subgraph(MHA) + 6xSin + Subgraph(Less)
                                 ::testing::Values(2),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);

const std::vector<std::vector<ov::Shape>> inputShapesWOTranspose = {
        {{1, 12, 197, 64}, {1, 12, 64, 197}, {1, 12, 197, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeOnInputs, MHAWOTransposeOnInputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(4),  // Subgraph + 3xSin
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         MHA::getTestCaseName);


} // namespace
} // namespace snippets
} // namespace test
} // namespace ov