// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// //

#include <vector>
#include <ngraph/opsets/opset6.hpp>

#include "single_layer_tests/gather_elements.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset6;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
};

INSTANTIATE_TEST_CASE_P(smoke_set1, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2})),     // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2})),     // Indices shape
                            ::testing::ValuesIn(std::vector<int>({-1, 0, 1})),  // Axis
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_set2, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2, 1})),  // Data shape
                            ::testing::Values(std::vector<size_t>({4, 2, 1})),  // Indices shape
                            ::testing::ValuesIn(std::vector<int>({0, -3})),     // Axis
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_set3, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 5})),   // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 7})),   // Indices shape
                            ::testing::Values(3, -1),                               // Axis
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_set4, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({3, 2, 3, 8})),   // Data shape
                            ::testing::Values(std::vector<size_t>({2, 2, 3, 8})),   // Indices shape
                            ::testing::Values(0, -4),                               // Axis
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_set5, GatherElementsLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<size_t>({3, 2, 3, 4, 8})),   // Data shape
                            ::testing::Values(std::vector<size_t>({3, 2, 3, 5, 8})),   // Indices shape
                            ::testing::Values(3, -2),                                  // Axis
                            ::testing::ValuesIn(inputPrecisions),
                            ::testing::ValuesIn(idxPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherElementsLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> ShapesRank4Axis0 = {
        std::vector<size_t>{1, 7, 8, 4},
        std::vector<size_t>{2, 7, 8, 4},
        std::vector<size_t>{7, 7, 8, 4},
        std::vector<size_t>{9, 7, 8, 4},
};
const std::vector<std::vector<size_t>> ShapesRank4Axis1 = {
        std::vector<size_t>{6, 1, 8, 4},
        std::vector<size_t>{6, 5, 8, 4},
        std::vector<size_t>{6, 8, 8, 4},
        std::vector<size_t>{6, 9, 8, 4},
};
const std::vector<std::vector<size_t>> ShapesRank4Axis2 = {
        std::vector<size_t>{6, 7, 2, 4},
        std::vector<size_t>{6, 7, 4, 4},
        std::vector<size_t>{6, 7, 5, 4},
        std::vector<size_t>{6, 7, 7, 4},
};
const std::vector<std::vector<size_t>> ShapesRank4Axis3 = {
        std::vector<size_t>{6, 5, 8, 1},
        std::vector<size_t>{6, 5, 8, 4},
        std::vector<size_t>{6, 5, 8, 7},
        std::vector<size_t>{6, 5, 8, 9},
};

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank4axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank4Axis0),              // Data shapes
        ::testing::ValuesIn(ShapesRank4Axis0),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 0 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank4axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank4Axis1),              // Data shapes
        ::testing::ValuesIn(ShapesRank4Axis1),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 1, -3 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank4axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank4Axis2),              // Data shapes
        ::testing::ValuesIn(ShapesRank4Axis2),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 2, -2 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank4axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank4Axis3),              // Data shapes
        ::testing::ValuesIn(ShapesRank4Axis3),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 3, -1 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> ShapesRank5Axis0 = {
        std::vector<size_t>{2, 3, 9, 4, 9},
        std::vector<size_t>{1, 3, 9, 4, 9},
        std::vector<size_t>{5, 3, 9, 4, 9},
        std::vector<size_t>{7, 3, 9, 4, 9},
};
const std::vector<std::vector<size_t>> ShapesRank5Axis1 = {
        std::vector<size_t>{2, 1, 5, 4, 7},
        std::vector<size_t>{2, 3, 5, 4, 7},
        std::vector<size_t>{2, 8, 5, 4, 7},
        std::vector<size_t>{2, 9, 5, 4, 7},
};
const std::vector<std::vector<size_t>> ShapesRank5Axis2 = {
        std::vector<size_t>{1, 2, 2, 8, 9},
        std::vector<size_t>{1, 2, 3, 8, 9},
        std::vector<size_t>{1, 2, 6, 8, 9},
        std::vector<size_t>{1, 2, 7, 8, 9},
};
const std::vector<std::vector<size_t>> ShapesRank5Axis3 = {
        std::vector<size_t>{2, 2, 4, 3, 7},
        std::vector<size_t>{2, 2, 4, 4, 7},
        std::vector<size_t>{2, 2, 4, 7, 7},
        std::vector<size_t>{2, 2, 4, 9, 7},
};
const std::vector<std::vector<size_t>> ShapesRank5Axis4 = {
        std::vector<size_t>{1, 3, 9, 3, 1},
        std::vector<size_t>{1, 3, 9, 3, 2},
        std::vector<size_t>{1, 3, 9, 3, 5},
        std::vector<size_t>{1, 3, 9, 3, 9},
};

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank5axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank5Axis0),              // Data shapes
        ::testing::ValuesIn(ShapesRank5Axis0),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 0 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank5axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank5Axis1),              // Data shapes
        ::testing::ValuesIn(ShapesRank5Axis1),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 1, -4 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank5axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank5Axis2),              // Data shapes
        ::testing::ValuesIn(ShapesRank5Axis2),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 2, -3 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank5axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank5Axis3),              // Data shapes
        ::testing::ValuesIn(ShapesRank5Axis3),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 3, -2 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank5axis4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank5Axis4),              // Data shapes
        ::testing::ValuesIn(ShapesRank5Axis4),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 4, -1 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> ShapesRank6Axis0 = {
        std::vector<size_t>{1, 3, 2, 4, 4, 3},
        std::vector<size_t>{3, 3, 2, 4, 4, 3},
        std::vector<size_t>{6, 3, 2, 4, 4, 3},
        std::vector<size_t>{7, 3, 2, 4, 4, 3},
};
const std::vector<std::vector<size_t>> ShapesRank6Axis1 = {
        std::vector<size_t>{1, 2, 2, 3, 5, 9},
        std::vector<size_t>{1, 5, 2, 3, 5, 9},
        std::vector<size_t>{1, 6, 2, 3, 5, 9},
        std::vector<size_t>{1, 9, 2, 3, 5, 9},
};
const std::vector<std::vector<size_t>> ShapesRank6Axis2 = {
        std::vector<size_t>{2, 3, 2, 7, 2, 1},
        std::vector<size_t>{2, 3, 5, 7, 2, 1},
        std::vector<size_t>{2, 3, 8, 7, 2, 1},
        std::vector<size_t>{2, 3, 9, 7, 2, 1},
};
const std::vector<std::vector<size_t>> ShapesRank6Axis3 = {
        std::vector<size_t>{1, 3, 4, 2, 1, 3},
        std::vector<size_t>{1, 3, 4, 4, 1, 3},
        std::vector<size_t>{1, 3, 4, 5, 1, 3},
        std::vector<size_t>{1, 3, 4, 8, 1, 3},
};
const std::vector<std::vector<size_t>> ShapesRank6Axis4 = {
        std::vector<size_t>{1, 3, 2, 4, 1, 3},
        std::vector<size_t>{1, 3, 2, 4, 4, 3},
        std::vector<size_t>{1, 3, 2, 4, 6, 3},
        std::vector<size_t>{1, 3, 2, 4, 7, 3},
};
const std::vector<std::vector<size_t>> ShapesRank6Axis5 = {
        std::vector<size_t>{2, 1, 7, 8, 1, 2},
        std::vector<size_t>{2, 1, 7, 8, 1, 3},
        std::vector<size_t>{2, 1, 7, 8, 1, 4},
        std::vector<size_t>{2, 1, 7, 8, 1, 6},
};

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank6axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank6Axis0),              // Data shapes
        ::testing::ValuesIn(ShapesRank6Axis0),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 0 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank6axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank6Axis1),              // Data shapes
        ::testing::ValuesIn(ShapesRank6Axis1),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 1, -5 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank6axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank6Axis2),              // Data shapes
        ::testing::ValuesIn(ShapesRank6Axis2),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 2, -4 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank6axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank6Axis3),              // Data shapes
        ::testing::ValuesIn(ShapesRank6Axis3),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 3, -3 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank6axis4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank6Axis4),              // Data shapes
        ::testing::ValuesIn(ShapesRank6Axis4),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 4, -2 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GatherElements_rank6axis5, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(ShapesRank6Axis5),              // Data shapes
        ::testing::ValuesIn(ShapesRank6Axis5),              // Indices shpae
        ::testing::ValuesIn(std::vector<int>({ 5, -1 })),
        ::testing::ValuesIn(inputPrecisions),               // Data precision
        ::testing::ValuesIn(idxPrecisions),                 // Indices precision
        ::testing::Values(CommonTestUtils::DEVICE_GPU)),    // Device name
    GatherElementsLayerTest::getTestCaseName);

}  // namespace
