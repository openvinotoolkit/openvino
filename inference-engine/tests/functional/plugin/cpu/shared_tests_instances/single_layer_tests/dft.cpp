// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/dft.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ngraph::helpers::DFTOpType> opTypes = {
    ngraph::helpers::DFTOpType::FORWARD,
    ngraph::helpers::DFTOpType::INVERSE
};

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::BF16
};

const std::vector<std::vector<size_t>> inputShapes = {
    {10, 4, 20, 32, 2},
    {2, 5, 7, 8, 2},
    {1, 120, 128, 1, 2},
};

/* 1D DFT */
const std::vector<std::vector<int64_t>> axes1D = {
    {0}, {1}, {2}, {3}, {-2}
};

const std::vector<std::vector<int64_t>> signalSizes1D = {
    {}, {16}, {40}
};

const auto testCase1D = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(axes1D),
    ::testing::ValuesIn(signalSizes1D),
    ::testing::ValuesIn(opTypes),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

/* 2D DFT */

const std::vector<std::vector<int64_t>> axes2D = {
    {0, 1}, {2, 1}, {2, 3}, {2, 0}, {1, 3}, {-1, -2}
};
const std::vector<std::vector<int64_t>> signalSizes2D = {
    {}, {5, 7}, {4, 10}, {16, 8}
};

const auto testCase2D = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(axes2D),
    ::testing::ValuesIn(signalSizes2D),
    ::testing::ValuesIn(opTypes),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


/* 3D DFT */

const std::vector<std::vector<int64_t>> axes3D = {
    {0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {2, 3, 1}, {-3, -1, -2},
};

const std::vector<std::vector<int64_t>> signalSizes3D = {
    {}, {4, 8, 16}, {7, 11, 32}
};

const auto testCase3D = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(axes3D),
    ::testing::ValuesIn(signalSizes3D),
    ::testing::ValuesIn(opTypes),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

/* 4D DFT */

const std::vector<std::vector<int64_t>> axes4D = {
    {0, 1, 2, 3}, {-1, 2, 0, 1}
};

const std::vector<std::vector<int64_t>> signalSizes4D = {
    {}, {5, 2, 5, 2}
};

const auto testCase4D = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(axes4D),
    ::testing::ValuesIn(signalSizes4D),
    ::testing::ValuesIn(opTypes),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


INSTANTIATE_TEST_CASE_P(smoke_MKLDNN_TestsDFT_1d, DFTLayerTest, testCase1D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_MKLDNN_TestsDFT_2d, DFTLayerTest, testCase2D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_MKLDNN_TestsDFT_3d, DFTLayerTest, testCase3D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_MKLDNN_TestsDFT_4d, DFTLayerTest, testCase4D, DFTLayerTest::getTestCaseName);
