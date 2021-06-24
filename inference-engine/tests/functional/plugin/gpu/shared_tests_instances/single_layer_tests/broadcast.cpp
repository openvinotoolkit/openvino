// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/broadcast.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::BOOL
};

// NUMPY MODE

std::vector<std::vector<size_t>> inShapesNumpy = {
        {3, 1},
        {1, 4, 1}
};

std::vector<std::vector<size_t>> targetShapesNumpy = {
        {2, 3, 6},
        {1, 4, 4}
};

const auto numpyBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesNumpy[0]),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(inShapesNumpy[0]),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestNumpyBroadcast1,
        BroadcastLayerTest,
        numpyBroadcastParams1,
        BroadcastLayerTest::getTestCaseName
);

const auto numpyBroadcastParams2 = ::testing::Combine(
        ::testing::Values(targetShapesNumpy[1]),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::Values(inShapesNumpy[1]),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestNumpyBroadcast2,
        BroadcastLayerTest,
        numpyBroadcastParams2,
        BroadcastLayerTest::getTestCaseName
);

// BIDIRECTIONAL MODE

std::vector<std::vector<size_t>> inShapesBidi = {
        {4, 1},
        {1, 4, 1},
        {4, 1, 1}
};

std::vector<std::vector<size_t>> targetShapesBidi = {
        {2, 1, 4},
        {1, 4, 4},
        {1, 1, 2, 2}
};

const auto bidirectionalBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[0]),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(inShapesBidi[0]),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestBidirectionalBroadcast1,
        BroadcastLayerTest,
        bidirectionalBroadcastParams1,
        BroadcastLayerTest::getTestCaseName
);

const auto bidirectionalBroadcastParams2 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[1]),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(inShapesBidi[1]),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestBidirectionalBroadcast2,
        BroadcastLayerTest,
        bidirectionalBroadcastParams2,
        BroadcastLayerTest::getTestCaseName
);

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

std::vector<ngraph::AxisSet> axes = {
        {1, 2},
        {0, 2}
};

const auto explicitBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesExplicit[0]),
        ::testing::Values(axes[0]),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
        ::testing::Values(inShapesExplicit[0]),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestExplicitBroadcast1,
        BroadcastLayerTest,
        explicitBroadcastParams1,
        BroadcastLayerTest::getTestCaseName
);

const auto explicitBroadcastParams2 = ::testing::Combine(
        ::testing::Values(targetShapesExplicit[1]),
        ::testing::Values(axes[1]),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
        ::testing::Values(inShapesExplicit[1]),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestExplicitBroadcast2,
        BroadcastLayerTest,
        explicitBroadcastParams2,
        BroadcastLayerTest::getTestCaseName
);
}  // namespace
