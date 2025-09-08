// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GatherLayerTest;
using ov::test::Gather7LayerTest;
using ov::test::Gather8LayerTest;
using ov::test::gather7ParamsTuple;
using ov::test::Gather8IndiceScalarLayerTest;
using ov::test::Gather8withIndicesDataLayerTest;

const std::vector<ov::element::Type> netPrecisionsFP32 = {
        ov::element::f32,
};

const std::vector<ov::element::Type> netPrecisionsI32 = {
        ov::element::i32,
};

const std::vector<ov::element::Type> netPrecisionsFP16 = {
        ov::element::f16,
};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::Shape> indicesShapes2 = {
        {2, 2},
        {2, 2, 2},
        {2, 4},
};

const std::vector<ov::Shape> indicesShapes23 = {
        {2, 3, 2},
        {2, 3, 4},
};

const std::vector<std::tuple<int, int>> axis_batch41 = {
        std::tuple<int, int>(3, 1),
        std::tuple<int, int>(4, 1),
};

const std::vector<std::tuple<int, int>> axis_batch42 = {
        std::tuple<int, int>(3, 2),
        std::tuple<int, int>(4, 2),
};

const std::vector<std::vector<ov::Shape>> inputShapesAxes4b1 = {
        {{2, 6, 7, 8, 9}},
        {{2, 1, 7, 8, 9}},
        {{2, 1, 1, 8, 9}},
        {{2, 6, 1, 4, 9}},
        {{2, 6, 7, 4, 1}},
        {{2, 6, 1, 8, 9}},
        {{2, 1, 7, 1, 9}},
        {{2, 6, 1, 8, 4}},
        {{2, 6, 7, 4, 9}},
        {{2, 1, 7, 8, 4}},
        {{2, 6, 7, 8, 4}},
};

const std::vector<std::vector<ov::Shape>> inputShapesAxes4b2 = {
        {{2, 3, 7, 8, 9}},
        {{2, 3, 7, 6, 9}},
        {{2, 3, 9, 8, 9}},
        {{2, 3, 9, 4, 9}},
        {{2, 3, 7, 4, 2}},
        {{2, 3, 5, 8, 9}},
        {{2, 3, 7, 2, 9}},
        {{2, 3, 9, 8, 4}},
        {{2, 3, 7, 4, 9}},
        {{2, 3, 7, 5, 4}},
        {{2, 3, 7, 8, 4}},
};

const auto GatherIndiceScalar = []() {
    return testing::Combine(testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{1, 3, 4, 5}}))),
                            testing::Values(ov::Shape({})),
                            testing::Values(std::tuple<int, int>(2, 0)),
                            testing::Values(ov::element::f32),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto GatherAxes4i4b1 = []() {
    return testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes4b1)),
                            testing::ValuesIn(indicesShapes2),
                            testing::ValuesIn(axis_batch41),
                            testing::ValuesIn(netPrecisions),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto GatherAxes4i8b1 = []() {
    return testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes4b1)),
                            testing::ValuesIn(indicesShapes2),
                            testing::ValuesIn(axis_batch41),
                            testing::ValuesIn(netPrecisions),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto GatherAxes4i8b2 = []() {
    return testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes4b2)),
                            testing::ValuesIn(indicesShapes23),
                            testing::ValuesIn(axis_batch42),
                            testing::ValuesIn(netPrecisions),
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
        {0, 3, 2, 1},
};
const std::vector<ov::Shape> indicesShapes12 = {
        {4},
        {2, 2}
};

const std::vector<ov::Shape> indicesShapes1 = {
        {4},
};

const std::vector<std::vector<ov::Shape>> inputShapesAxes4 = {
        {{5, 6, 7, 8, 9}},
        {{1, 6, 7, 8, 9}},
        {{5, 1, 7, 8, 9}},
        {{5, 6, 1, 8, 9}},
        {{5, 6, 7, 1, 9}},
};

const std::vector<std::vector<ov::Shape>> inputShapes6DAxes4 = {
        {{5, 6, 7, 8, 9, 10}},
        {{1, 1, 7, 8, 9, 10}},
        {{5, 1, 1, 8, 9, 10}},
        {{5, 6, 1, 1, 9, 10}},
        {{5, 6, 7, 1, 9, 1}},
        {{1, 6, 1, 8, 9, 10}},
        {{5, 1, 7, 1, 9, 10}},
        {{5, 6, 1, 8, 9, 1}},
        {{1, 6, 7, 1, 9, 10}},
        {{5, 1, 7, 8, 9, 1}},
        {{1, 6, 7, 8, 9, 1}},
};

const std::vector<int> axes4 = {4};

const auto GatherAxes4 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes4),
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes4)),
                            testing::Values(ov::element::f16),
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
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes6DAxes4)),
                            testing::Values(ov::element::f32),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes4,
        GatherLayerTest,
        Gather6dAxes4(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<ov::Shape>> inputShapesAxes3 = {
        {{5, 6, 7, 8}},
        {{1, 6, 7, 8}},
        {{5, 1, 7, 8}},
        {{5, 6, 1, 8}},
        {{5, 6, 7, 8, 9}},
        {{1, 6, 7, 8, 9}},
        {{5, 1, 7, 8, 9}},
        {{5, 6, 1, 8, 9}},
        {{5, 6, 7, 8, 1}},
};

const std::vector<std::vector<ov::Shape>> inputShapes6DAxes3 = {
        {{5, 6, 7, 8, 9, 10}},
        {{1, 1, 7, 8, 9, 10}},
        {{5, 1, 1, 8, 9, 10}},
        {{5, 6, 1, 8, 1, 10}},
        {{5, 6, 7, 8, 1, 1}},
        {{1, 6, 1, 8, 9, 10}},
        {{5, 1, 7, 8, 1, 10}},
        {{5, 6, 1, 8, 9, 1}},
        {{1, 6, 7, 8, 1, 10}},
        {{5, 1, 7, 8, 9, 1}},
        {{1, 6, 7, 8, 9, 1}},
};

const std::vector<int> axes3 = {3};

const auto GatherAxes3 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes3),
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes3)),
                            testing::Values(ov::element::f32),
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
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes6DAxes3)),
                            testing::Values(ov::element::i32),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes3,
        GatherLayerTest,
        Gather6dAxes3(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<ov::Shape>> inputShapesAxes2 = {
        {{5, 6, 7}},
        {{5, 6, 7, 8}},
        {{1, 6, 7, 8}},
        {{5, 1, 7, 8}},
        {{5, 6, 7, 1}},
        {{5, 6, 7, 8, 9}},
        {{1, 6, 7, 8, 9}},
        {{5, 1, 7, 8, 9}},
        {{5, 6, 7, 1, 9}},
        {{5, 6, 7, 8, 1}},
};

const std::vector<std::vector<ov::Shape>> inputShapes6DAxes2 = {
        {{5, 6, 7, 8, 9, 10}},
        {{1, 1, 7, 8, 9, 10}},
        {{5, 1, 7, 1, 9, 10}},
        {{5, 6, 7, 1, 1, 10}},
        {{5, 6, 7, 8, 1, 1}},
        {{1, 6, 7, 1, 9, 10}},
        {{5, 1, 7, 8, 1, 10}},
        {{5, 6, 7, 1, 9, 1}},
        {{1, 6, 7, 8, 1, 10}},
        {{5, 1, 7, 8, 9, 1}},
        {{1, 6, 7, 8, 9, 1}},
};

const std::vector<int> axes2 = {2};

const auto GatherAxes2 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes2),
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes2)),
                            testing::Values(ov::element::f32),
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
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes6DAxes2)),
                            testing::Values(ov::element::f16),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes2,
        GatherLayerTest,
        Gather6dAxes2(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<ov::Shape>> inputShapesAxes1 = {
        {{5, 6}},
        {{5, 6, 7}},
        {{5, 6, 7, 8}},
        {{1, 6, 7, 8}},
        {{5, 6, 1, 8}},
        {{5, 6, 7, 1}},
        {{5, 6, 7, 8, 9}},
        {{1, 6, 7, 8, 9}},
        {{5, 6, 1, 8, 9}},
        {{5, 6, 7, 1, 9}},
        {{5, 6, 7, 8, 1}},
};

const std::vector<std::vector<ov::Shape>> inputShapes6DAxes1 = {
        {{5, 6, 7, 8, 9, 10}},
        {{1, 6, 1, 8, 9, 10}},
        {{5, 6, 1, 1, 9, 10}},
        {{5, 6, 7, 1, 1, 10}},
        {{5, 6, 7, 8, 1, 1}},
        {{1, 6, 7, 1, 9, 10}},
        {{5, 6, 1, 8, 1, 10}},
        {{5, 6, 1, 8, 9, 1}},
        {{1, 6, 7, 8, 1, 10}},
        {{1, 6, 7, 8, 9, 1}},
        {{5, 6, 7, 1, 9, 1}},
};

const std::vector<int> axes1 = {1};

const auto GatherAxes1 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes1),
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes1)),
                            testing::Values(ov::element::i32),
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
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes6DAxes1)),
                            testing::Values(ov::element::f32),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes1,
        GatherLayerTest,
        Gather6dAxes1(),
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<ov::Shape>> inputShapesAxes0 = {
        {{5}},
        {{5, 6}},
        {{5, 6, 7}},
        {{5, 6, 7, 8}},
        {{5, 1, 7, 8}},
        {{5, 6, 1, 8}},
        {{5, 6, 7, 1}},
        {{5, 6, 7, 8, 9}},
        {{5, 1, 7, 8, 9}},
        {{5, 6, 1, 8, 9}},
        {{5, 6, 7, 1, 9}},
        {{5, 6, 7, 8, 1}},
};

const std::vector<std::vector<ov::Shape>> inputShapes6DAxes0 = {
        {{5, 6, 7, 8, 9, 10}},
        {{5, 1, 1, 8, 9, 10}},
        {{5, 6, 1, 1, 9, 10}},
        {{5, 6, 7, 1, 1, 10}},
        {{5, 6, 7, 8, 1, 1}},
        {{5, 1, 7, 1, 9, 10}},
        {{5, 6, 1, 8, 1, 10}},
        {{5, 6, 1, 8, 9, 1}},
        {{5, 1, 7, 8, 1, 10}},
        {{5, 1, 7, 8, 9, 1}},
        {{5, 6, 7, 1, 9, 1}},
};

const std::vector<int> axes0 = {0};

const auto GatherAxes0 = []() {
    return testing::Combine(testing::ValuesIn(indices),
                            testing::ValuesIn(indicesShapes12),
                            testing::ValuesIn(axes0),
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesAxes0)),
                            testing::Values(ov::element::f32),
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
                            testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes6DAxes0)),
                            testing::Values(ov::element::f32),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather6dAxes0,
        GatherLayerTest,
        Gather6dAxes0(),
        GatherLayerTest::getTestCaseName
);

const auto GatherAxes0Optimized = []() {
    return testing::Combine(testing::ValuesIn(ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{4, 8, 2, 2}}}))),
                            testing::Values(ov::Shape({})),
                            testing::Values(std::tuple<int, int>(0, 0)),
                            testing::Values(ov::element::f32),
                            testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Gather7Axes0Optimized,
        Gather8IndiceScalarLayerTest,
        GatherAxes0Optimized(),
        Gather8IndiceScalarLayerTest::getTestCaseName
);

gather7ParamsTuple dummyParams = {
        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{2, 3}})),
        ov::Shape({2, 2}),
        std::tuple<int, int>{1, 1},
        ov::element::f32,
        ov::test::utils::DEVICE_GPU,
};

std::vector<std::vector<int64_t>> indicesData = {
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

INSTANTIATE_TEST_SUITE_P(smoke,
        Gather8withIndicesDataLayerTest,
        gatherWithIndicesParams,
        Gather8withIndicesDataLayerTest::getTestCaseName
);

std::vector<std::vector<int64_t>> nagativeSingleindicesData = {
        {-1},
        {-2},
        {-3}
};

gather7ParamsTuple dummyParams2 = {
        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{4, 8, 2, 2}})),
        ov::Shape({}),
        std::tuple<int, int>{0, 0},
        ov::element::f32,
        ov::test::utils::DEVICE_GPU,
};

const auto gatherWithNagativeIndicesParams1 = testing::Combine(
        testing::Values(dummyParams2),
        testing::ValuesIn(nagativeSingleindicesData)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather8NagativeIndice1,
        Gather8withIndicesDataLayerTest,
        gatherWithNagativeIndicesParams1,
        Gather8withIndicesDataLayerTest::getTestCaseName
);

gather7ParamsTuple dummyParams3 = {
        ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{6, 8, 2, 2}})),
        ov::Shape({}),
        std::tuple<int, int>{0, 0},
        ov::element::f32,
        ov::test::utils::DEVICE_GPU,
};

const auto gatherWithNagativeIndicesParams2 = testing::Combine(
        testing::Values(dummyParams3),
        testing::ValuesIn(nagativeSingleindicesData)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather8NagativeIndice2,
        Gather8withIndicesDataLayerTest,
        gatherWithNagativeIndicesParams2,
        Gather8withIndicesDataLayerTest::getTestCaseName
);

}  // namespace
