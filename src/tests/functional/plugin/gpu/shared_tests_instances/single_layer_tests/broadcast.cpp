// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/broadcast.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> inputTPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::BOOL
};

// NUMPY MODE //////////////////////////////////////////
// 1D
std::vector<std::vector<size_t>> targetShapesNumpy1D = {
        {1},
        {16},
        {1, 16},
        {1, 1, 16},
        {2, 16},
        {2, 3, 6},
        {1, 4, 4},
};

const auto numpyBroadcast1DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesNumpy1D),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(std::vector<size_t>{1}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast1D, BroadcastLayerTest, numpyBroadcast1DInputParams, BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_PrecTransformation, BroadcastLayerTest,
        ::testing::Combine(
            ::testing::Values(targetShapesNumpy1D[0]),
            ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
            ::testing::Values(ngraph::op::BroadcastType::NUMPY),
            ::testing::Values(std::vector<size_t>{1}),
            ::testing::ValuesIn(inputTPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
        BroadcastLayerTest::getTestCaseName);

// 2D
std::vector<std::vector<size_t>> targetShapesNumpy2D = {
        {3, 1},
        {3, 6},
        {2, 3, 6},
        {2, 2, 3, 6},
};

const auto numpyBroadcast2DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesNumpy2D),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(std::vector<size_t>{3, 1}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast2D, BroadcastLayerTest, numpyBroadcast2DInputParams, BroadcastLayerTest::getTestCaseName);

// 3D
std::vector<std::vector<size_t>> targetShapesNumpy3D = {
        {1, 4, 1},
        {1, 4, 4},
        {1, 1, 4, 4},
        {2, 1, 1, 4, 4},
};

const auto numpyBroadcast3DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesNumpy3D),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(std::vector<size_t>{1, 4, 1}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast3D, BroadcastLayerTest, numpyBroadcast3DInputParams, BroadcastLayerTest::getTestCaseName);

const auto numpyBroadcastParams6D = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 2, 3, 4, 5, 6}),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(std::vector<size_t>{1, 2, 1, 4, 1, 6}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);
INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast6D, BroadcastLayerTest, numpyBroadcastParams6D, BroadcastLayerTest::getTestCaseName);

const auto numpyBroadcastParams5D = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 2, 3, 4, 5}),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(std::vector<size_t>{1, 2, 1, 4, 1}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestNumpyBroadcast5D, BroadcastLayerTest, numpyBroadcastParams5D, BroadcastLayerTest::getTestCaseName);
// END NUMPY MODE //////////////////////////////////////

// BIDIRECTIONAL MODE //////////////////////////////////
std::vector<std::vector<size_t>> inShapesBidi = {
        {4, 1},
        {1, 4, 1},
        {4, 1, 1}
};

std::vector<std::vector<size_t>> targetShapesBidi = {
        {4, 1, 4},
        {1, 4, 4},
        {1, 1, 4, 4}
};

const auto bidirectionalBroadcastParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesBidi),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::ValuesIn(inShapesBidi),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestBidirectionalBroadcast, BroadcastLayerTest, bidirectionalBroadcastParams, BroadcastLayerTest::getTestCaseName);

// EXPLICIT MODE ///////////////////////////////////////
// 1D
std::vector<std::vector<size_t>> inShapesExplicit1D = { {4} };
std::vector<std::vector<size_t>> targetShapesExplicit1D = { {4, 2, 4}, {4, 2, 4, 1} };
std::vector<ngraph::AxisSet> axes1D = { {0}, {2} };

const auto explicitBroadcast1DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesExplicit1D),
        ::testing::ValuesIn(axes1D),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inShapesExplicit1D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast1D, BroadcastLayerTest, explicitBroadcast1DInputParams, BroadcastLayerTest::getTestCaseName);

const auto bidirectionalBroadcastParams3 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[2]),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(inShapesBidi[2]),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestBidirectionalBroadcast3,
        BroadcastLayerTest,
        bidirectionalBroadcastParams3,
        BroadcastLayerTest::getTestCaseName
);

// EXPLICIT MODE

std::vector<std::vector<size_t>> inShapesExplicit = {
        {3, 1},
        {2, 4}
};

std::vector<std::vector<size_t>> targetShapesExplicit = {
        {2, 3, 1},
        {2, 3, 4}
};

// 2D
std::vector<std::vector<size_t>> inShapesExplicit2D = { {2, 4} };
std::vector<std::vector<size_t>> targetShapesExplicit2D = { {2, 2, 4}, {2, 2, 4, 1}};
std::vector<ngraph::AxisSet> axes2D = { {1, 2}, {0, 2} };

const auto explicitBroadcast2DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesExplicit2D),
        ::testing::ValuesIn(axes2D),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inShapesExplicit2D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast2D, BroadcastLayerTest, explicitBroadcast2DInputParams, BroadcastLayerTest::getTestCaseName);

// 3D
std::vector<std::vector<size_t>> inShapesExplicit3D = { {2, 2, 2} };
std::vector<std::vector<size_t>> targetShapesExplicit3D = { {2, 2, 2, 2} };
std::vector<ngraph::AxisSet> axes3D = { {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3} };

const auto explicitBroadcast3DInputParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesExplicit3D),
        ::testing::ValuesIn(axes3D),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inShapesExplicit3D),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestExplicitBroadcast3D, BroadcastLayerTest, explicitBroadcast3DInputParams, BroadcastLayerTest::getTestCaseName);
// END EXPLICIT MODE ///////////////////////////////////

}  // namespace
