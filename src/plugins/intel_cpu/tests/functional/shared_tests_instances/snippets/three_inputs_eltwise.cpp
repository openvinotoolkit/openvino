// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/three_inputs_eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

    INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, ThreeInputsEltwise,
                         ::testing::Combine(
                                 ::testing::Values(ov::Shape {1, 64, 10, 10}),
                                 ::testing::Values(ov::Shape {1, 64, 10,  1}),
                                 ::testing::Values(ov::Shape {1, 1, 1,  10}),
                                 ::testing::Values(2), // eltwises fuse only for non-broadcasted shapes
                                 ::testing::Values(0), // SnippetsMarkSkipped disables tokenization for eltwise chains after inputs
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ThreeInputsEltwise::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, ThreeInputsEltwiseSinh,
            ::testing::Combine(
            ::testing::Values(ov::Shape {1, 64, 10, 10}),
            ::testing::Values(ov::Shape {1, 64, 10,  1}),
            ::testing::Values(ov::Shape {1, 1, 1,  10}),
            ::testing::Values(4), // Subgraph + 3 converts after inputs
            ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             ThreeInputsEltwiseSinh::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov