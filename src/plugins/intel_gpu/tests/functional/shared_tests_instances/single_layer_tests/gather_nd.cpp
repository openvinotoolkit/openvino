// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/gather_nd.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GatherNDLayerTest;
using ov::test::GatherND8LayerTest;

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> idxPrecisions = {
        ov::element::i32,
        ov::element::i64,
};

// set1
std::vector<std::vector<ov::Shape>> shapes_subset1_static = std::vector<std::vector<ov::Shape>>({
    {{2, 2}}, {{2, 3, 4}} });
std::vector<ov::Shape> indices_subset1_static = std::vector<ov::Shape>({
    {2, 1}, {2, 1, 1}});

// set2
std::vector<std::vector<ov::Shape>> shapes_subset2_static = std::vector<std::vector<ov::Shape>>({
    {{15, 12, 20, 15, 2}}, {{15, 12, 18, 7, 17}}});
std::vector<ov::Shape> indices_subset2_static = std::vector<ov::Shape>({
    {15, 12, 2}, {15, 12, 5, 9, 1, 3}});

// set3
std::vector<std::vector<ov::Shape>> shapes_subset3_static = std::vector<std::vector<ov::Shape>>({
    {{4, 3, 2, 5, 5, 2}}, {{4, 3, 2, 5, 7, 2}} });
std::vector<ov::Shape> indices_subset3_static = std::vector<ov::Shape>({
    {4, 3, 2, 5, 1}, {4, 3, 2, 5, 6, 2}});


// -------------------------------- V5 --------------------------------
INSTANTIATE_TEST_SUITE_P(smoke_GatherND5_set1, GatherNDLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset1_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>(indices_subset1_static)),
        ::testing::ValuesIn(std::vector<int>({ 0, 1 })),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherNDLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND5_set2, GatherNDLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset2_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>(indices_subset2_static)),
        ::testing::ValuesIn(std::vector<int>({ 1, 2 })),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherNDLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND5_set3, GatherNDLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset3_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>(indices_subset3_static)),
        ::testing::ValuesIn(std::vector<int>({ 3, 4 })),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherNDLayerTest::getTestCaseName);

// -------------------------------- V8 --------------------------------
INSTANTIATE_TEST_SUITE_P(smoke_GatherND8_set1, GatherND8LayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset1_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>(indices_subset1_static)),
        ::testing::ValuesIn(std::vector<int>({ 0, 1 })),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND8_set2, GatherND8LayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset2_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>(indices_subset2_static)),
        ::testing::ValuesIn(std::vector<int>({ 1, 2 })),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND8_set3, GatherND8LayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_subset3_static)),
        ::testing::ValuesIn(std::vector<ov::Shape>(indices_subset3_static)),
        ::testing::ValuesIn(std::vector<int>({ 3, 4 })),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherND8LayerTest::getTestCaseName);

}  // namespace
