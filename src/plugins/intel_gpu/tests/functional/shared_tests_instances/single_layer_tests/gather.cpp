// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisionsFP32 = {
        InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> netPrecisionsI32 = {
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> netPrecisionsFP16 = {
        InferenceEngine::Precision::FP16,
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const std::vector<std::vector<size_t>> indicesShapes2 = {
        std::vector<size_t>{2, 2},
        std::vector<size_t>{2, 2, 2},
        std::vector<size_t>{2, 4},
};

const std::vector<std::vector<size_t>> indicesShapes23 = {
        std::vector<size_t>{2, 3, 2},
        std::vector<size_t>{2, 3, 4},
};

const std::vector<std::tuple<int, int>> axis_batch41 = {
        std::tuple<int, int>(3, 1),
        std::tuple<int, int>(4, 1),
};

const std::vector<std::tuple<int, int>> axis_batch42 = {
        std::tuple<int, int>(3, 2),
        std::tuple<int, int>(4, 2),
};

const std::vector<std::vector<size_t>> inputShapesAxes4b1 = {
        std::vector<size_t>{2, 6, 7, 8, 9},
        std::vector<size_t>{2, 1, 7, 8, 9},
        std::vector<size_t>{2, 1, 1, 8, 9},
        std::vector<size_t>{2, 6, 1, 4, 9},
        std::vector<size_t>{2, 6, 7, 4, 1},
        std::vector<size_t>{2, 6, 1, 8, 9},
        std::vector<size_t>{2, 1, 7, 1, 9},
        std::vector<size_t>{2, 6, 1, 8, 4},
        std::vector<size_t>{2, 6, 7, 4, 9},
        std::vector<size_t>{2, 1, 7, 8, 4},
        std::vector<size_t>{2, 6, 7, 8, 4},
};

const std::vector<std::vector<size_t>> inputShapesAxes4b2 = {
        std::vector<size_t>{2, 3, 7, 8, 9},
        std::vector<size_t>{2, 3, 7, 6, 9},
        std::vector<size_t>{2, 3, 9, 8, 9},
        std::vector<size_t>{2, 3, 9, 4, 9},
        std::vector<size_t>{2, 3, 7, 4, 2},
        std::vector<size_t>{2, 3, 5, 8, 9},
        std::vector<size_t>{2, 3, 7, 2, 9},
        std::vector<size_t>{2, 3, 9, 8, 4},
        std::vector<size_t>{2, 3, 7, 4, 9},
        std::vector<size_t>{2, 3, 7, 5, 4},
        std::vector<size_t>{2, 3, 7, 8, 4},
};

const auto GatherIndiceScalar = []() {
    return testing::Combine(testing::Values(std::vector<size_t>{1, 3, 4, 5}),
                            testing::Values(std::vector<size_t>{}),
                            testing::Values(std::tuple<int, int>(2, 0)),
                            testing::Values(InferenceEngine::Precision::FP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto GatherAxes4i4b1 = []() {
    return testing::Combine(testing::ValuesIn(inputShapesAxes4b1),
                            testing::ValuesIn(indicesShapes2),
                            testing::ValuesIn(axis_batch41),
                            testing::ValuesIn(netPrecisions),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto GatherAxes4i8b1 = []() {
    return testing::Combine(testing::ValuesIn(inputShapesAxes4b1),
                            testing::ValuesIn(indicesShapes2),
                            testing::ValuesIn(axis_batch41),
                            testing::ValuesIn(netPrecisions),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto GatherAxes4i8b2 = []() {
    return testing::Combine(testing::ValuesIn(inputShapesAxes4b2),
                            testing::ValuesIn(indicesShapes23),
                            testing::ValuesIn(axis_batch42),
                            testing::ValuesIn(netPrecisions),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i4b1,
        Gather7LayerTest,
        GatherAxes4i4b1(),
        Gather7LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i4b2,
        Gather7LayerTest,
        GatherAxes4i4b1(),
        Gather7LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i8b1,
        Gather7LayerTest,
        GatherAxes4i8b1(),
        Gather7LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i8b2,
        Gather7LayerTest,
        GatherAxes4i8b2(),
        Gather7LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i4b1,
        Gather8LayerTest,
        GatherAxes4i4b1(),
        Gather8LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i4b2,
        Gather8LayerTest,
        GatherAxes4i4b1(),
        Gather8LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i8b1,
        Gather8LayerTest,
        GatherAxes4i8b1(),
        Gather8LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes4i8b2,
        Gather8LayerTest,
        GatherAxes4i8b2(),
        Gather8LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_GatherIndiceScalar,
        Gather8IndiceScalarLayerTest,
        GatherIndiceScalar(),
        Gather8IndiceScalarLayerTest::getTestCaseName
);

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes12 = {
        std::vector<size_t>{4},
        std::vector<size_t>{2, 2}
};

const std::vector<std::vector<size_t>> indicesShapes1 = {
        std::vector<size_t>{4},
};

const std::vector<std::vector<size_t>> inputShapes6DAxes5 = {
        std::vector<size_t>{5, 6, 7, 8, 9, 10},
        std::vector<size_t>{1, 1, 7, 8, 9, 10},
        std::vector<size_t>{5, 1, 1, 8, 9, 10},
        std::vector<size_t>{5, 6, 1, 1, 9, 10},
        std::vector<size_t>{5, 6, 7, 1, 1, 10},
        std::vector<size_t>{1, 6, 1, 8, 9, 10},
        std::vector<size_t>{5, 1, 7, 1, 9, 10},
        std::vector<size_t>{5, 6, 1, 8, 1, 10},
        std::vector<size_t>{1, 6, 7, 1, 9, 10},
        std::vector<size_t>{5, 1, 7, 8, 1, 10},
        std::vector<size_t>{1, 6, 7, 8, 1, 10},
};

const std::vector<int> axes5 = {5};

const std::vector<std::vector<size_t>> inputShapesAxes4 = {
        std::vector<size_t>{5, 6, 7, 8, 9},
        std::vector<size_t>{1, 6, 7, 8, 9},
        std::vector<size_t>{5, 1, 7, 8, 9},
        std::vector<size_t>{5, 6, 1, 8, 9},
        std::vector<size_t>{5, 6, 7, 1, 9},
};

const std::vector<std::vector<size_t>> inputShapes6DAxes4 = {
        std::vector<size_t>{5, 6, 7, 8, 9, 10},
        std::vector<size_t>{1, 1, 7, 8, 9, 10},
        std::vector<size_t>{5, 1, 1, 8, 9, 10},
        std::vector<size_t>{5, 6, 1, 1, 9, 10},
        std::vector<size_t>{5, 6, 7, 1, 9, 1},
        std::vector<size_t>{1, 6, 1, 8, 9, 10},
        std::vector<size_t>{5, 1, 7, 1, 9, 10},
        std::vector<size_t>{5, 6, 1, 8, 9, 1},
        std::vector<size_t>{1, 6, 7, 1, 9, 10},
        std::vector<size_t>{5, 1, 7, 8, 9, 1},
        std::vector<size_t>{1, 6, 7, 8, 9, 1},
};

const std::vector<int> axes4 = {4};

const auto GatherAxes4 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes4),
                            testing::ValuesIn(inputShapesAxes4),
                            testing::ValuesIn(netPrecisionsFP16),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_GatherAxes4,
        GatherLayerTest,
        GatherAxes4(),
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes4 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes1),
                            testing::ValuesIn(axes4),
                            testing::ValuesIn(inputShapes6DAxes4),
                            testing::ValuesIn(netPrecisionsFP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes4,
        GatherLayerTest,
        Gather6dAxes4(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesAxes3 = {
        std::vector<size_t>{5, 6, 7, 8},
        std::vector<size_t>{1, 6, 7, 8},
        std::vector<size_t>{5, 1, 7, 8},
        std::vector<size_t>{5, 6, 1, 8},
        std::vector<size_t>{5, 6, 7, 8, 9},
        std::vector<size_t>{1, 6, 7, 8, 9},
        std::vector<size_t>{5, 1, 7, 8, 9},
        std::vector<size_t>{5, 6, 1, 8, 9},
        std::vector<size_t>{5, 6, 7, 8, 1},
};

const std::vector<std::vector<size_t>> inputShapes6DAxes3 = {
        std::vector<size_t>{5, 6, 7, 8, 9, 10},
        std::vector<size_t>{1, 1, 7, 8, 9, 10},
        std::vector<size_t>{5, 1, 1, 8, 9, 10},
        std::vector<size_t>{5, 6, 1, 8, 1, 10},
        std::vector<size_t>{5, 6, 7, 8, 1, 1},
        std::vector<size_t>{1, 6, 1, 8, 9, 10},
        std::vector<size_t>{5, 1, 7, 8, 1, 10},
        std::vector<size_t>{5, 6, 1, 8, 9, 1},
        std::vector<size_t>{1, 6, 7, 8, 1, 10},
        std::vector<size_t>{5, 1, 7, 8, 9, 1},
        std::vector<size_t>{1, 6, 7, 8, 9, 1},
};

const std::vector<int> axes3 = {3};

const auto GatherAxes3 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes3),
                            testing::ValuesIn(inputShapesAxes3),
                            testing::ValuesIn(netPrecisionsFP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_GatherAxes3,
        GatherLayerTest,
        GatherAxes3(),
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes3 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes1),
                            testing::ValuesIn(axes3),
                            testing::ValuesIn(inputShapes6DAxes3),
                            testing::ValuesIn(netPrecisionsI32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes3,
        GatherLayerTest,
        Gather6dAxes3(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesAxes2 = {
        std::vector<size_t>{5, 6, 7},
        std::vector<size_t>{5, 6, 7, 8},
        std::vector<size_t>{1, 6, 7, 8},
        std::vector<size_t>{5, 1, 7, 8},
        std::vector<size_t>{5, 6, 7, 1},
        std::vector<size_t>{5, 6, 7, 8, 9},
        std::vector<size_t>{1, 6, 7, 8, 9},
        std::vector<size_t>{5, 1, 7, 8, 9},
        std::vector<size_t>{5, 6, 7, 1, 9},
        std::vector<size_t>{5, 6, 7, 8, 1},
};

const std::vector<std::vector<size_t>> inputShapes6DAxes2 = {
        std::vector<size_t>{5, 6, 7, 8, 9, 10},
        std::vector<size_t>{1, 1, 7, 8, 9, 10},
        std::vector<size_t>{5, 1, 7, 1, 9, 10},
        std::vector<size_t>{5, 6, 7, 1, 1, 10},
        std::vector<size_t>{5, 6, 7, 8, 1, 1},
        std::vector<size_t>{1, 6, 7, 1, 9, 10},
        std::vector<size_t>{5, 1, 7, 8, 1, 10},
        std::vector<size_t>{5, 6, 7, 1, 9, 1},
        std::vector<size_t>{1, 6, 7, 8, 1, 10},
        std::vector<size_t>{5, 1, 7, 8, 9, 1},
        std::vector<size_t>{1, 6, 7, 8, 9, 1},
};

const std::vector<int> axes2 = {2};

const auto GatherAxes2 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes2),
                            testing::ValuesIn(inputShapesAxes2),
                            testing::ValuesIn(netPrecisionsFP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_GatherAxes2,
        GatherLayerTest,
        GatherAxes2(),
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes2 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes1),
                            testing::ValuesIn(axes2),
                            testing::ValuesIn(inputShapes6DAxes2),
                            testing::ValuesIn(netPrecisionsFP16),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes2,
        GatherLayerTest,
        Gather6dAxes2(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesAxes1 = {
        std::vector<size_t>{5, 6},
        std::vector<size_t>{5, 6, 7},
        std::vector<size_t>{5, 6, 7, 8},
        std::vector<size_t>{1, 6, 7, 8},
        std::vector<size_t>{5, 6, 1, 8},
        std::vector<size_t>{5, 6, 7, 1},
        std::vector<size_t>{5, 6, 7, 8, 9},
        std::vector<size_t>{1, 6, 7, 8, 9},
        std::vector<size_t>{5, 6, 1, 8, 9},
        std::vector<size_t>{5, 6, 7, 1, 9},
        std::vector<size_t>{5, 6, 7, 8, 1},
};

const std::vector<std::vector<size_t>> inputShapes6DAxes1 = {
        std::vector<size_t>{5, 6, 7, 8, 9, 10},
        std::vector<size_t>{1, 6, 1, 8, 9, 10},
        std::vector<size_t>{5, 6, 1, 1, 9, 10},
        std::vector<size_t>{5, 6, 7, 1, 1, 10},
        std::vector<size_t>{5, 6, 7, 8, 1, 1},
        std::vector<size_t>{1, 6, 7, 1, 9, 10},
        std::vector<size_t>{5, 6, 1, 8, 1, 10},
        std::vector<size_t>{5, 6, 1, 8, 9, 1},
        std::vector<size_t>{1, 6, 7, 8, 1, 10},
        std::vector<size_t>{1, 6, 7, 8, 9, 1},
        std::vector<size_t>{5, 6, 7, 1, 9, 1},
};

const std::vector<int> axes1 = {1};

const auto GatherAxes1 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes1),
                            testing::ValuesIn(inputShapesAxes1),
                            testing::ValuesIn(netPrecisionsI32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_GatherAxes1,
        GatherLayerTest,
        GatherAxes1(),
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes1 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes1),
                            testing::ValuesIn(axes1),
                            testing::ValuesIn(inputShapes6DAxes1),
                            testing::ValuesIn(netPrecisionsFP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes1,
        GatherLayerTest,
        Gather6dAxes1(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesAxes0 = {
        std::vector<size_t>{5},
        std::vector<size_t>{5, 6},
        std::vector<size_t>{5, 6, 7},
        std::vector<size_t>{5, 6, 7, 8},
        std::vector<size_t>{5, 1, 7, 8},
        std::vector<size_t>{5, 6, 1, 8},
        std::vector<size_t>{5, 6, 7, 1},
        std::vector<size_t>{5, 6, 7, 8, 9},
        std::vector<size_t>{5, 1, 7, 8, 9},
        std::vector<size_t>{5, 6, 1, 8, 9},
        std::vector<size_t>{5, 6, 7, 1, 9},
        std::vector<size_t>{5, 6, 7, 8, 1},
};

const std::vector<std::vector<size_t>> inputShapes6DAxes0 = {
        std::vector<size_t>{5, 6, 7, 8, 9, 10},
        std::vector<size_t>{5, 1, 1, 8, 9, 10},
        std::vector<size_t>{5, 6, 1, 1, 9, 10},
        std::vector<size_t>{5, 6, 7, 1, 1, 10},
        std::vector<size_t>{5, 6, 7, 8, 1, 1},
        std::vector<size_t>{5, 1, 7, 1, 9, 10},
        std::vector<size_t>{5, 6, 1, 8, 1, 10},
        std::vector<size_t>{5, 6, 1, 8, 9, 1},
        std::vector<size_t>{5, 1, 7, 8, 1, 10},
        std::vector<size_t>{5, 1, 7, 8, 9, 1},
        std::vector<size_t>{5, 6, 7, 1, 9, 1},
};

const std::vector<int> axes0 = {0};

const auto GatherAxes0 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes0),
                            testing::ValuesIn(inputShapesAxes0),
                            testing::ValuesIn(netPrecisionsFP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_GatherAxes0,
        GatherLayerTest,
        GatherAxes0(),
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes0 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes1),
                            testing::ValuesIn(axes0),
                            testing::ValuesIn(inputShapes6DAxes0),
                            testing::ValuesIn(netPrecisionsFP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes0,
        GatherLayerTest,
        Gather6dAxes0(),
        GatherLayerTest::getTestCaseName
);

const auto GatherAxes0Optimized = []() {
    return testing::Combine(testing::ValuesIn({std::vector<size_t>{4, 8, 2, 2}}),
                            testing::ValuesIn({std::vector<size_t>{}}),
                            testing::ValuesIn({std::tuple<int, int>{0, 0}}),
                            testing::ValuesIn(netPrecisionsFP32),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(InferenceEngine::Layout::ANY),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes0Optimized,
        Gather8IndiceScalarLayerTest,
        GatherAxes0Optimized(),
        Gather8IndiceScalarLayerTest::getTestCaseName
);

gather7ParamsTuple dummyParams = {
        std::vector<size_t>{2, 3},
        std::vector<size_t>{2, 2},
        std::tuple<int, int>{1, 1},
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::UNSPECIFIED,
        InferenceEngine::Precision::UNSPECIFIED,
        InferenceEngine::Layout::ANY,
        InferenceEngine::Layout::ANY,
        ov::test::utils::DEVICE_GPU,
};

std::vector<std::vector<int>> indicesData = {
        {0, 1, 2, 0},           // positive in bound
        {-1, -2, -3, -1},       // negative in bound
        {-1, 0, 1, 2},          // positive and negative in bound
        {0, 1, 2, 3},           // positive out of bound
        {-1, -2, -3, -4},       // negative out of bound
        {0, 4, -4, 0},          // positive and negative out of bound
};

const auto gatherWithIndicesParams = testing::Combine(
        testing::Values(dummyParams),
        testing::ValuesIn(indicesData)
);

INSTANTIATE_TEST_CASE_P(smoke, Gather8withIndicesDataLayerTest, gatherWithIndicesParams, Gather8withIndicesDataLayerTest::getTestCaseName);

}  // namespace
