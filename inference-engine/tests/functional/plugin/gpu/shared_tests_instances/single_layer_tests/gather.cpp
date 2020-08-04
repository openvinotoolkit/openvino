// Copyright (C) 2019 Intel Corporation
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

const auto Gather6dAxes5 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1),
        testing::ValuesIn(axes5),
        testing::ValuesIn(inputShapes6DAxes5),
        testing::ValuesIn(netPrecisionsFP32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

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

const auto GatherAxes4 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes12),
        testing::ValuesIn(axes4),
        testing::ValuesIn(inputShapesAxes4),
        testing::ValuesIn(netPrecisionsFP16),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherAxes4,
        GatherLayerTest,
        GatherAxes4,
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes4 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1),
        testing::ValuesIn(axes4),
        testing::ValuesIn(inputShapes6DAxes4),
        testing::ValuesIn(netPrecisionsFP32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        Gather6dAxes4,
        GatherLayerTest,
        Gather6dAxes4,
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

const auto GatherAxes3 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes12),
        testing::ValuesIn(axes3),
        testing::ValuesIn(inputShapesAxes3),
        testing::ValuesIn(netPrecisionsFP32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherAxes3,
        GatherLayerTest,
        GatherAxes3,
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes3 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1),
        testing::ValuesIn(axes3),
        testing::ValuesIn(inputShapes6DAxes3),
        testing::ValuesIn(netPrecisionsI32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        Gather6dAxes3,
        GatherLayerTest,
        Gather6dAxes3,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesAxes2 = {
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

const auto GatherAxes2 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes12),
        testing::ValuesIn(axes2),
        testing::ValuesIn(inputShapesAxes2),
        testing::ValuesIn(netPrecisionsFP32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherAxes2,
        GatherLayerTest,
        GatherAxes2,
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes2 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1),
        testing::ValuesIn(axes2),
        testing::ValuesIn(inputShapes6DAxes2),
        testing::ValuesIn(netPrecisionsFP16),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        Gather6dAxes2,
        GatherLayerTest,
        Gather6dAxes2,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesAxes1 = {
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

const auto GatherAxes1 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes12),
        testing::ValuesIn(axes1),
        testing::ValuesIn(inputShapesAxes1),
        testing::ValuesIn(netPrecisionsI32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherAxes1,
        GatherLayerTest,
        GatherAxes1,
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes1 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1),
        testing::ValuesIn(axes1),
        testing::ValuesIn(inputShapes6DAxes1),
        testing::ValuesIn(netPrecisionsFP32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        Gather6dAxes1,
        GatherLayerTest,
        Gather6dAxes1,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesAxes0 = {
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

const auto GatherAxes0 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes12),
        testing::ValuesIn(axes0),
        testing::ValuesIn(inputShapesAxes0),
        testing::ValuesIn(netPrecisionsFP32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherAxes0,
        GatherLayerTest,
        GatherAxes0,
        GatherLayerTest::getTestCaseName
);

const auto Gather6dAxes0 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1),
        testing::ValuesIn(axes0),
        testing::ValuesIn(inputShapes6DAxes0),
        testing::ValuesIn(netPrecisionsFP32),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        Gather6dAxes0,
        GatherLayerTest,
        Gather6dAxes0,
        GatherLayerTest::getTestCaseName
);

}  // namespace
