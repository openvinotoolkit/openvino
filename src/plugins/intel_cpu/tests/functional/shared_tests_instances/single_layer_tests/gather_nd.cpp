// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/gather_nd.hpp"


namespace {
using ov::test::GatherNDLayerTest;
using ov::test::GatherND8LayerTest;

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
        ov::element::i64,
        ov::element::i16,
        ov::element::u8,
        ov::element::i8
};
const std::vector<ov::element::Type> indices_types = {
        ov::element::i32,
        ov::element::i64
};

std::vector<std::vector<ov::Shape>> shapes_subset1_static = {{{2, 2}}, {{2, 3, 4}}};

const auto gatherNDArgsSubset1 = ::testing::Combine(
);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND5_Set1, GatherNDLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset1_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>({{2, 1}, {2, 1, 1}})),
        ::testing::ValuesIn(std::vector<int>({0, 1})),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        GatherNDLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND8_Set1, GatherND8LayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset1_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>({{2, 1}, {2, 1, 1}})),
        ::testing::ValuesIn(std::vector<int>({0, 1})),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        GatherNDLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> shapes_subset2_static = {{{15, 12, 20, 15, 2}}, {{15, 12, 18, 7, 17}}};

INSTANTIATE_TEST_SUITE_P(smoke_GatherND5_Set2, GatherNDLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset2_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>{{15, 12, 2}, {15, 12, 5, 9, 1, 3}}),
        ::testing::ValuesIn(std::vector<int>({0, 1, 2})),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        GatherNDLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND8_Set2, GatherND8LayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset2_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>{{15, 12, 2}, {15, 12, 5, 9, 1, 3}}),
        ::testing::ValuesIn(std::vector<int>({0, 1, 2})),
        ::testing::ValuesIn(model_types),
        ::testing::ValuesIn(indices_types),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        GatherNDLayerTest::getTestCaseName);

}  // namespace
