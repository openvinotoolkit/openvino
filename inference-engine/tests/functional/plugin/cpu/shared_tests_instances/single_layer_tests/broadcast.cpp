// Copyright (C) 2019 Intel Corporation
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
        //InferenceEngine::Precision::U8, TODO: not work for bidirectional
        //InferenceEngine::Precision::BOOL
};

// NUMPY MODE

std::vector<std::vector<size_t>> inShapesNumpy = {
        {3, 1}
};

std::vector<std::vector<size_t>> targetShapesNumpy = {
        {2, 3, 6}
};

const auto numpyBroadcastParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesNumpy),
        ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
        ::testing::ValuesIn(inShapesNumpy),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        TestNumpyBroadcast,
        BroadcastLayerTest,
        numpyBroadcastParams,
        BroadcastLayerTest::getTestCaseName
);

// BIDIRECTIONAL MODE

std::vector<std::vector<size_t>> inShapesBidi = {
        {4, 1}
};

std::vector<std::vector<size_t>> targetShapesBidi = {
        {2, 1, 4}
};

const auto bidirectionalBroadcastParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesBidi),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::ValuesIn(inShapesBidi),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        TestBidirectionalBroadcast,
        BroadcastLayerTest,
        bidirectionalBroadcastParams,
        BroadcastLayerTest::getTestCaseName
);

// EXPLICIT MODE

std::vector<std::vector<size_t>> inShapesExplicit = {
        {3, 1}
};

std::vector<std::vector<size_t>> targetShapesExplicit = {
        {2, 3, 1}
};

std::vector<ngraph::AxisSet> axes = {
        {1, 2}
};

const auto explicitBroadcastParams = ::testing::Combine(
        ::testing::ValuesIn(targetShapesExplicit),
        ::testing::ValuesIn(axes),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
        ::testing::ValuesIn(inShapesExplicit),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        TestExplicitBroadcast,
        BroadcastLayerTest,
        explicitBroadcastParams,
        BroadcastLayerTest::getTestCaseName
);
}  // namespace