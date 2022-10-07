// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Add,
                     ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 42, 16, 64}),
                             ::testing::Values(ov::Shape {1, 42, 16,  1}),
                             ::testing::Values(1), // one node - Add
                             ::testing::Values(0), // SnippetsMarkSkipped disables tokenization for eltwise chains after inputs
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     Add::getTestCaseName);


namespace snippets_static_1 {
// These  inputs are needed to test static TileScheduler optimizations (emit the whole tile, body with increments, set WA etc)
//std::vector<ov::Shape> inShapesStatic1{{1, 16, 29,  1}, {1, 16, 29,  7}, {1, 16, 29,  8}, {1, 16, 29,  15}, {1, 16, 29,  16}, {1, 16, 29,  31}};
//std::vector<ov::Shape> inShapesStatic2{{1, 16, 29,  1}, {1, 16, 1, 1}, {1, 1, 1, 1}};
std::vector<ov::Shape> inShapesStatic1{{1, 16, 29,  7}, };
std::vector<ov::Shape> inShapesStatic2{{1, 16, 29,  1}, };
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddSinh,
                         ::testing::Combine(
                             ::testing::ValuesIn(inShapesStatic1),
                             ::testing::ValuesIn(inShapesStatic2),
                             ::testing::Values(3), // Add + 2 converts after inputs
                             ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         AddSinh::getTestCaseName);

} // namespace snippets_static_1

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddSinhConst,
                     ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 42, 16, 64}),
                             ::testing::Values(2), // Add + 2 converts after inputs
                             ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     AddSinhConst::getTestCaseName);
}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov