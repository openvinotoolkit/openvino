// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/select.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

//============================Select=======================================//
std::vector<ov::test::InputShape> inShapes_a{{{}, {{1, 5, 5, 35}}}};
std::vector<ov::test::InputShape> inShapes_b{{{}, {{1}}}};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Select, Select,
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_a),
                ::testing::ValuesIn(inShapes_a),
                ::testing::ValuesIn(inShapes_b),
                ::testing::ValuesIn({ov::element::f32, ov::element::i8}),
                ::testing::Values(1),
                ::testing::Values(1),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        Select::getTestCaseName);

// DS
std::vector<ov::test::InputShape> inShapesDynamic_a{{{1, {1, 5}, -1, 35}, {{1, 5, 5, 35}, {1, 1, 1, 35}, {1, 5, 5, 35}}}};
std::vector<ov::test::InputShape> inShapesDynamic_b{{{-1}, {{1}, {1}, {1}}}};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Select_Dynamic, Select,
        ::testing::Combine(
                ::testing::ValuesIn(inShapesDynamic_a),
                ::testing::ValuesIn(inShapesDynamic_a),
                ::testing::ValuesIn(inShapesDynamic_b),
                ::testing::ValuesIn({ov::element::f32, ov::element::i8}),
                ::testing::Values(1),
                ::testing::Values(1),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        Select::getTestCaseName);

//============================BroadcastSelect=======================================//
std::vector<ov::test::InputShape> inShapes0{{{}, {{1, 8, 2, 1}}}, {{}, {{1, 1, 1, 1}}}};
std::vector<ov::test::InputShape> inShapes1{{{}, {{1, 8, 2, 10}}}, {{}, {{1, 8, 2, 1}}}};
std::vector<ov::test::InputShape> inShapes2{{{}, {{1, 8, 2, 10}}}, {{}, {{1, 1, 1, 1}}}};
std::vector<ov::PartialShape> inShapes3{{1, 8, 2, 1}, {1, 8, 2, 10}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastSelect, BroadcastSelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes0),
                                 ::testing::ValuesIn(inShapes1),
                                 ::testing::ValuesIn(inShapes2),
                                 ::testing::ValuesIn(inShapes3),
                                 ::testing::ValuesIn({ov::element::f32, ov::element::i8}),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         BroadcastSelect::getTestCaseName);

// DS
std::vector<ov::test::InputShape> inShapes0_d{{{-1, -1, -1, -1}, {{1, 8, 2, 1}, {1, 1, 1, 1}, {1, 8, 2, 1}}}};
std::vector<ov::test::InputShape> inShapes1_d{{{1, -1, -1, -1}, {{1, 8, 2, 10}, {1, 8, 2, 10}, {1, 8, 2, 10}}}};
std::vector<ov::test::InputShape> inShapes2_d{{{1, {1, 8}, {1, 2}, {1, 10}}, {{1, 8, 2, 10}, {1, 1, 2, 1}, {1, 8, 2, 10}}}};
std::vector<ov::PartialShape> inShapes3_d{{1, 8, 2, 1}, {1, 8, 2, 10}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastSelect_Dynamic, BroadcastSelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes0_d),
                                 ::testing::ValuesIn(inShapes1_d),
                                 ::testing::ValuesIn(inShapes2_d),
                                 ::testing::ValuesIn(inShapes3_d),
                                 ::testing::ValuesIn({ov::element::f32, ov::element::i8}),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         BroadcastSelect::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov