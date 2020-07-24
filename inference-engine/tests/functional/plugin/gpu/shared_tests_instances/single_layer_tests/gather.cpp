// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
};

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes = {
        std::vector<size_t>{4}
        // 5d output not supported yet
        // std::vector<size_t>{2, 2}
};

const std::vector<int> axes = {0, 1, 2, 3};


const auto params = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        Gather,
        GatherLayerTest,
        params,
        GatherLayerTest::getTestCaseName
);

// TODO: remove additional bfyx WA tests below and add 5d/6d test cases to cartesian product above
// when proper support of any inputs of that kind will be added to clDNN Gather primitive

const std::vector<std::vector<size_t>> indicesShapes1d = {
        std::vector<size_t>{4}
};

const std::vector<std::vector<size_t>> indicesShapes2d = {
        std::vector<size_t>{2, 2}
};

// For axes 0, 1 and 2 we can provide pretty much any input, as dimensions after 'axis'
// will be merged, so we will end up in bfyx format anyway

const std::vector<std::vector<size_t>> inputShapesWAInd1dAxis012 = {
        std::vector<size_t>{5, 6, 7, 8, 9},
        std::vector<size_t>{5, 6, 7, 8, 9, 10},
};

const auto paramsWAInd1dAxis012 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1d),
        testing::ValuesIn({0, 1, 2}),
        testing::ValuesIn(inputShapesWAInd1dAxis012),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherBfyxWaInd1dAxis012,
        GatherLayerTest,
        paramsWAInd1dAxis012,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesWAInd2dAxis012 = {
        std::vector<size_t>{5, 6, 7, 8},
        std::vector<size_t>{5, 6, 7, 8, 9},
};

const auto paramsWAInd2dAxis012 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes2d),
        testing::ValuesIn({0, 1, 2}),
        testing::ValuesIn(inputShapesWAInd2dAxis012),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherBfyxWaInd2dAxis012,
        GatherLayerTest,
        paramsWAInd2dAxis012,
        GatherLayerTest::getTestCaseName
);

// For axes 3, 4 and 5 we can still support some inputs, as long as
// they have enough unit dimensions to convert whole operation to bfyx format

const std::vector<std::vector<size_t>> inputShapesWAInd1dAxis3 = {
        std::vector<size_t>{1, 6, 7, 8, 9},
        std::vector<size_t>{5, 1, 7, 8, 9},
        std::vector<size_t>{5, 6, 1, 8, 9},
        std::vector<size_t>{5, 6, 7, 8, 1},
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

const auto paramsWAInd1dAxis3 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1d),
        testing::ValuesIn({3}),
        testing::ValuesIn(inputShapesWAInd1dAxis3),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherBfyxWaInd1dAxis3,
        GatherLayerTest,
        paramsWAInd1dAxis3,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesWAInd2dAxis3 = {
        std::vector<size_t>{1, 6, 7, 8},
        std::vector<size_t>{5, 1, 7, 8},
        std::vector<size_t>{5, 6, 1, 8},
        std::vector<size_t>{1, 1, 7, 8, 9},
        std::vector<size_t>{5, 1, 1, 8, 9},
        std::vector<size_t>{5, 6, 1, 8, 1},
        std::vector<size_t>{1, 6, 1, 8, 9},
        std::vector<size_t>{5, 1, 7, 8, 1},
        std::vector<size_t>{1, 6, 7, 8, 1},
};

const auto paramsWAInd2dAxis3 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes2d),
        testing::ValuesIn({3}),
        testing::ValuesIn(inputShapesWAInd2dAxis3),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherBfyxWaInd2dAxis3,
        GatherLayerTest,
        paramsWAInd2dAxis3,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesWAInd1dAxis4 = {
        std::vector<size_t>{1, 6, 7, 8, 9},
        std::vector<size_t>{5, 1, 7, 8, 9},
        std::vector<size_t>{5, 6, 1, 8, 9},
        std::vector<size_t>{5, 6, 7, 1, 9},
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

const auto paramsWAInd1dAxis4 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1d),
        testing::ValuesIn({4}),
        testing::ValuesIn(inputShapesWAInd1dAxis4),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherBfyxWaInd1dAxis4,
        GatherLayerTest,
        paramsWAInd1dAxis4,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesWAInd2dAxis4 = {
        std::vector<size_t>{1, 1, 7, 8, 9},
        std::vector<size_t>{5, 1, 1, 8, 9},
        std::vector<size_t>{5, 6, 1, 1, 9},
        std::vector<size_t>{1, 6, 1, 8, 9},
        std::vector<size_t>{5, 1, 7, 1, 9},
        std::vector<size_t>{1, 6, 7, 1, 9},
};

const auto paramsWAInd2dAxis4 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes2d),
        testing::ValuesIn({4}),
        testing::ValuesIn(inputShapesWAInd2dAxis4),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherBfyxWaInd2dAxis4,
        GatherLayerTest,
        paramsWAInd2dAxis4,
        GatherLayerTest::getTestCaseName
);

const std::vector<std::vector<size_t>> inputShapesWAInd1dAxis5 = {
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

const auto paramsWAInd1dAxis5 = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes1d),
        testing::ValuesIn({5}),
        testing::ValuesIn(inputShapesWAInd1dAxis5),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(
        GatherBfyxWaInd1dAxis5,
        GatherLayerTest,
        paramsWAInd1dAxis5,
        GatherLayerTest::getTestCaseName
);

}  // namespace
