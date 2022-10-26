// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather.hpp"
#include "ngraph_functions/builders.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

// Just need to check types transformation.
const std::vector<InferenceEngine::Precision> netPrecisionsTrCheck = {
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes_1D = {
        std::vector<size_t>{4},
};

const std::vector<std::vector<size_t>> indicesShapes_1D = {
        std::vector<size_t>{1},
        std::vector<size_t>{3},
};

const std::vector<std::tuple<int, int>> axes_batchdims_1D = {
        std::tuple<int, int>{0, 0}
};

const auto gather7Params_1D = testing::Combine(
        testing::ValuesIn(inputShapes_1D),
        testing::ValuesIn(indicesShapes_1D),
        testing::ValuesIn(axes_batchdims_1D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_1D, Gather7LayerTest, gather7Params_1D, Gather7LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TypesTrf, Gather7LayerTest,
            testing::Combine(
                testing::ValuesIn(inputShapes_1D),
                testing::ValuesIn(indicesShapes_1D),
                testing::ValuesIn(axes_batchdims_1D),
                testing::ValuesIn(netPrecisionsTrCheck),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes_2D = {
        std::vector<size_t>{4, 19},
};

const std::vector<std::vector<size_t>> indicesShapes_2D = {
        std::vector<size_t>{4, 1},
        std::vector<size_t>{4, 2},
};

const std::vector<std::tuple<int, int>> axes_batchdims_2D = {
        std::tuple<int, int>{0, 0},
        std::tuple<int, int>{1, 0},
        std::tuple<int, int>{1, 1},
        std::tuple<int, int>{-1, -1},
};

const auto gather7Params_2D = testing::Combine(
        testing::ValuesIn(inputShapes_2D),
        testing::ValuesIn(indicesShapes_2D),
        testing::ValuesIn(axes_batchdims_2D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_2D, Gather7LayerTest, gather7Params_2D, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes4D = {
        std::vector<size_t>{4, 5, 6, 7},
};

const std::vector<std::vector<size_t>> indicesShapes_BD0 = {
        std::vector<size_t>{4},
        std::vector<size_t>{2, 2},
        std::vector<size_t>{3, 3},
        std::vector<size_t>{5, 2},
        std::vector<size_t>{3, 2, 4},
};

const std::vector<std::tuple<int, int>> axes_BD0 = {
        std::tuple<int, int>{0, 0},
        std::tuple<int, int>{1, 0},
        std::tuple<int, int>{2, 0},
        std::tuple<int, int>{-1, 0},
};

const auto gather7ParamsSubset_BD0 = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_BD0),
        testing::ValuesIn(axes_BD0),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_BD0, Gather7LayerTest, gather7ParamsSubset_BD0, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_BD0, Gather8LayerTest, gather7ParamsSubset_BD0, Gather8LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> indicesShapes_BD1 = {
        std::vector<size_t>{4, 2},
        std::vector<size_t>{4, 5, 3},
        std::vector<size_t>{4, 1, 2, 3},
};

const std::vector<std::tuple<int, int>> axes_BD1 = {
        std::tuple<int, int>{1, 1},
        std::tuple<int, int>{2, 1},
        std::tuple<int, int>{-1, 1},
        std::tuple<int, int>{-2, 1},
};

const auto gather7ParamsSubset_BD1 = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_BD1),
        testing::ValuesIn(axes_BD1),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_BD1, Gather7LayerTest, gather7ParamsSubset_BD1, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> indicesShapes_BD2 = {
        std::vector<size_t>{4, 5, 4, 3},
        std::vector<size_t>{4, 5, 3, 2}
};

const std::vector<std::tuple<int, int>> axes_BD2 = {
        std::tuple<int, int>{2, 2},
        std::tuple<int, int>{3, -2},
        std::tuple<int, int>{-1, 2},
        std::tuple<int, int>{-1, -2},
};

const auto gather7ParamsSubset_BD2 = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_BD2),
        testing::ValuesIn(axes_BD2),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_BD2, Gather7LayerTest, gather7ParamsSubset_BD2, Gather7LayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> indicesShapes_NegativeBD = {
        std::vector<size_t>{4, 5, 4},
        std::vector<size_t>{4, 5, 3}
};

const std::vector<std::tuple<int, int>> axes_NegativeBD = {
        std::tuple<int, int>{0, -3},
        std::tuple<int, int>{1, -2},
        std::tuple<int, int>{2, -2},
        std::tuple<int, int>{-2, -2},
        std::tuple<int, int>{-1, -1},
        std::tuple<int, int>{-2, -1},
};

const auto gather7ParamsSubset_NegativeBD = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(indicesShapes_NegativeBD),
        testing::ValuesIn(axes_NegativeBD),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Gather7_NegativeBD, Gather7LayerTest, gather7ParamsSubset_NegativeBD, Gather7LayerTest::getTestCaseName);


///// GATHER-8 /////

const std::vector<std::vector<size_t>> dataShapes4DGather8 = {
        {10, 3, 1, 2},
        {10, 3, 3, 1},
        {10, 2, 2, 7},
        {10, 2, 2, 2},
        {10, 3, 4, 4},
        {10, 2, 3, 17}
};
const std::vector<std::vector<size_t>> idxShapes4DGather8 = {
        {10, 1, 1},
        {10, 1, 2},
        {10, 1, 3},
        {10, 2, 2},
        {10, 1, 7},
        {10, 2, 4},
        {10, 3, 3},
        {10, 3, 5},
        {10, 7, 3},
        {10, 8, 7}
};
const std::vector<std::tuple<int, int>> axesBatches4DGather8 = {
        {3, 0},
        {-1, -2},
        {2, -3},
        {2, 1},
        {1, 0},
        {1, 1},
        {0, 0}
};

INSTANTIATE_TEST_CASE_P(smoke_static_4D, Gather8LayerTest,
        testing::Combine(
                testing::ValuesIn(dataShapes4DGather8),
                testing::ValuesIn(idxShapes4DGather8),
                testing::ValuesIn(axesBatches4DGather8),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(CommonTestUtils::DEVICE_CPU)),
        Gather8LayerTest::getTestCaseName);

const auto gatherParamsVec2 = testing::Combine(
        testing::ValuesIn(std::vector<std::vector<size_t>>({{5, 4}, {11, 4}, {23, 4}, {35, 4}, {51, 4}, {71, 4}})),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{1}})),
        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{1, 0}}),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_Vec2, Gather8LayerTest, gatherParamsVec2, Gather8LayerTest::getTestCaseName);

const auto gatherParamsVec3 = testing::Combine(
        testing::ValuesIn(std::vector<std::vector<size_t>>({{4, 4}})),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{5}, {11}, {21}, {35}, {55}, {70}})),
        testing::ValuesIn(std::vector<std::tuple<int, int>>{std::tuple<int, int>{1, 0}}),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_Vec3, Gather8LayerTest, gatherParamsVec3, Gather8LayerTest::getTestCaseName);

}  // namespace
