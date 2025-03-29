// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {
// ===================================Add=========================================================//
// These  inputs are needed to test static Loop optimizations (emit the whole tile, body with increments, set WA etc)
std::vector<ov::test::InputShape> inShapesStatic1{{{}, {{1, 16, 29, 1}}},
                                                  {{}, {{1, 16, 29, 7}}},
                                                  {{}, {{1, 16, 29, 8}}},
                                                  {{}, {{1, 16, 29, 15}}},
                                                  {{}, {{1, 16, 29, 16}}},
                                                  {{}, {{1, 16, 29, 31}}}};
std::vector<ov::test::InputShape> inShapesStatic2{{{}, {{1, 16, 29, 1}}},
                                                  {{}, {{1, 16, 1, 1}}},
                                                  {{}, {{1, 1, 1, 1}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Add,
                         ::testing::Combine(
                             ::testing::ValuesIn(inShapesStatic1),
                             ::testing::ValuesIn(inShapesStatic2),
                             ::testing::ValuesIn({ov::element::f32, ov::element::f16}),
                             ::testing::Values(1), // Add
                             ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Add::getTestCaseName);

// DS
std::vector<InputShape> inShapesDynamic1{
        {
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 3, 1, 10}, {1, 3, 10, 10}, {1, 3, 1, 10}}},
        }
};
std::vector<InputShape> inShapesDynamic2{
        {
        {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        {{1, 3, 10, 1}, {1, 3, 1, 1}, {1, 3, 10, 1}}},
        }
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise_Add, Add,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapesDynamic1),
                                 ::testing::ValuesIn(inShapesDynamic2),
                                 ::testing::ValuesIn({ov::element::f32, ov::element::f16}),
                                 ::testing::Values(1),
                                 ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Add::getTestCaseName);

// ===================================AddPair=========================================================//
// test cross-tile (vector vs scalar) optimizations in the absence of vector tile
std::vector<std::vector<InputShape>> inShapesAddPair {
        {{{}, {{1, 128, 1, 1}}}, {{}, {{1, 128, 1, 1}}}},
        {{{}, {{1, 128, 1, 9}}}, {{}, {{1, 128, 1, 9}}}},
        {{{}, {{1, 128, 1, 16}}}, {{}, {{1, 128, 1, 16}}}},
        {{{}, {{1, 128, 1, 17}}}, {{}, {{1, 128, 1, 17}}}},
        {{{}, {{1, 128, 1, 29}}}, {{}, {{1, 128, 1, 29}}}},
        {{{}, {{1, 128, 1, 33}}}, {{}, {{1, 128, 1, 33}}}},
        {{{}, {{1, 128, 9, 30}}}, {{}, {{1, 128, 1, 30}}}},
        {{{}, {{1, 128, 9, 1}}}, {{}, {{1, 128, 1, 30}}}},
        {{{}, {{1, 128, 9, 16}}}, {{}, {{1, 128, 9, 1}}}},
        // Test Canonicalization and Dimension collapsing
        {{{}, {{2, 17, 3, 4}}}, {{}, {{1, 3, 4}}}},
        {{{}, {{2, 17, 3, 4}}}, {{}, {{1, 4}}}},
        {{{}, {{32, 5, 10}}}, {{}, {{5, 10}}}},
        {{{}, {{32, 5, 10}}}, {{}, {{10}}}},
        {{{}, {{30}}}, {{}, {{10, 30}}}},
        {{{}, {{5}}}, {{}, {{5}}}},
        // DS
        {{{1, -1, {1, 10}, {1, 33}}, {{1, 128, 1, 1}, {1, 128, 1, 9}, {1, 128, 1, 17}, {1, 128, 1, 29}, {1, 128, 9, 1}, {1, 128, 1, 1}}},
         {{{1, 1}, {128, 128}, {1, 10}, {1, 33}}, {{1, 128, 1, 1}, {1, 128, 1, 9}, {1, 128, 1, 17}, {1, 128, 1, 29}, {1, 128, 1, 30}, {1, 128, 1, 1}}}},
        {{{1, -1, 1, {1, 32}}, {{1, 16, 1, 32}, {1, 16, 1, 32}, {1, 16, 1, 32}, {1, 16, 1, 32}}},
         {{1, -1, 1, {1, 32}}, {{1, 16, 1, 32}, {1, 16, 1, 32}, {1, 16, 1, 32}, {1, 16, 1, 32}}}},
        {{{-1, 39}, {{1, 39}, {2, 39}, {1, 39}, {5, 39}, {2, 39}}},
         {{-1, 39}, {{1, 39}, {1, 39}, {10, 39}, {5, 39}, {1, 39}}}},
};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddPair,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapesAddPair),
                                 ::testing::ValuesIn({ov::element::f32, ov::element::f16}),
                                 ::testing::Values(1),
                                 ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         AddPair::getTestCaseName);

// ===================================AddConst, AddRollConst=========================================================//
std::vector<ov::test::InputShape> inShapesAddConst{{{}, {{1, 2, 3,  32}}},
                                                   {{}, {{1, 3, 17, 33}}},
                                                   {{-1, -1, -1, -1}, {{1, 3, 17, 33}, {1, 2, 1, 65}, {1, 3, 17, 33}}},
                                                   {{1, {1, 10}, {1, 8}, {1, 4}}, {{1, 2, 8, 4}, {1, 8, 1, 1}, {1, 2, 8, 4}}}};
std::vector<PartialShape> inShapesConstAddConst{{1, 1, 1, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddConst,
        ::testing::Combine(
                ::testing::ValuesIn(inShapesAddConst),
                ::testing::ValuesIn(inShapesConstAddConst),
                ::testing::Values(ov::element::f32),
                ::testing::Values(1), // Add
                ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        AddConst::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise_FP16, AddConst,
        ::testing::Combine(
                ::testing::ValuesIn(inShapesAddConst),
                ::testing::ValuesIn(inShapesConstAddConst),
                ::testing::Values(ov::element::f16),
                ::testing::Values(1), // Add
                ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        AddConst::getTestCaseName);
// ===================================AddRollConst=========================================================//
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddRollConst,
        ::testing::Combine(
                ::testing::ValuesIn(inShapesAddConst),
                ::testing::ValuesIn(inShapesConstAddConst),
                ::testing::Values(ov::element::f32),
                ::testing::Values(2), // Add + roll after inputs
                ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        AddRollConst::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise_BF16, AddRollConst,
        ::testing::Combine(
                ::testing::ValuesIn(inShapesAddConst),
                ::testing::ValuesIn(inShapesConstAddConst),
                ::testing::Values(ov::element::bf16),
                ::testing::Values(3), // Add + reorder + roll after inputs
                ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        AddRollConst::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov