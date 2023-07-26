// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

namespace snippets_static_1 {
// These  inputs are needed to test static Loop optimizations (emit the whole tile, body with increments, set WA etc)
std::vector<ov::Shape> inShapesStatic1{{1, 16, 29,  1}, {1, 16, 29,  7}, {1, 16, 29,  8}, {1, 16, 29,  15}, {1, 16, 29,  16}, {1, 16, 29,  31}};
std::vector<ov::Shape> inShapesStatic2{{1, 16, 29,  1}, {1, 16, 1, 1}, {1, 1, 1, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Add,
                         ::testing::Combine(
                             ::testing::ValuesIn(inShapesStatic1),
                             ::testing::ValuesIn(inShapesStatic2),
                             ::testing::Values(ov::element::f32),
                             ::testing::Values(1), // Add
                             ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Add::getTestCaseName);
// test cross-tile (vector vs scalar) optimizations in the absence of vector tile
std::vector<std::vector<ov::Shape>> inShapesStatic{
        {{1, 128, 1, 1}, {1, 128, 1, 1}},
        {{1, 128, 1, 9}, {1, 128, 1, 9}},
        {{1, 128, 1, 16}, {1, 128, 1, 16}},
        {{1, 128, 1, 17}, {1, 128, 1, 17}},
        {{1, 128, 1, 29}, {1, 128, 1, 29}},
        {{1, 128, 1, 33}, {1, 128, 1, 33}},
        {{1, 128, 9, 30}, {1, 128, 1, 30}},
        {{1, 128, 9, 1}, {1, 128, 1, 30}},
        {{1, 128, 9, 16}, {1, 128, 9, 1}},
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddPair,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapesStatic),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(1),
                                 ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         AddPair::getTestCaseName);

} // namespace snippets_static_1

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddConst,
        ::testing::Combine(
                ::testing::Values(ov::Shape {1, 42, 16, 64}),
                ::testing::Values(ov::element::f32),
                ::testing::Values(1), // Add
                ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        AddConst::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddRollConst,
        ::testing::Combine(
                ::testing::Values(ov::Shape {1, 42, 16, 64}),
                ::testing::Values(ov::element::f32),
                ::testing::Values(2), // Add + roll after inputs
                ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        AddRollConst::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise_BF16, AddRollConst,
        ::testing::Combine(
                ::testing::Values(ov::Shape {1, 2, 3, 32}),
                ::testing::Values(ov::element::bf16),
                ::testing::Values(3), // Add + reorder + roll after inputs
                ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        AddRollConst::getTestCaseName);
}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov