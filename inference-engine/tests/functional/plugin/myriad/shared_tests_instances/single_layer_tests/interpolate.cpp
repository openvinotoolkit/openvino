// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vpu/private_plugin_config.hpp>
#include "common/myriad_common_test_utils.hpp"
#include <vector>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

typedef std::map<std::string, std::string> Config;

const std::vector<std::vector<size_t>> inShapes = {
        {2, 8, 35, 35},
};

const std::vector<std::vector<size_t>> targetShapes = {
        {2, 8, 46, 46},
};

const std::vector<std::vector<size_t>> inShapesNoBatch = {
        {1, 8, 6, 6},
};

const std::vector<std::vector<size_t>> targetShapesNoBatch = {
        {1, 8, 12, 12},
};

const std::vector<std::vector<size_t>> inShapesNightly = {
        {2, 8, 38, 38},
        {2, 8, 3, 3},
        {2, 8, 4, 4},
        {2, 8, 16, 16},
        {2, 8, 31, 31},
        {2, 8, 26, 26},
};

const std::vector<std::vector<size_t>> targetShapesNightly = {
        {2, 8, 38 * 2, 38 * 2},
        {2, 8, 6, 6},
        {2, 8, 3, 3},
        {2, 8, 36, 36},
        {2, 8, 72, 72},
        {2, 8, 30, 30},
};

const std::vector<std::vector<size_t>> inShapes2x = {
        {2, 19, 37, 37},
};

const std::vector<std::vector<size_t>> targetShapes2x = {
        {2, 19, 37 * 2, 37 * 2},
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
        ngraph::op::v4::Interpolate::InterpolateMode::linear,
        ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::nearest,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearest = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearest2x = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesNearestMore = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModesWithoutNearest = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::simple,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
        ngraph::op::v4::Interpolate::NearestMode::floor,
        ngraph::op::v4::Interpolate::NearestMode::ceil,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestModeMore = {
        ngraph::op::v4::Interpolate::NearestMode::floor,
};

const std::vector<std::vector<size_t>> pads = {
        {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxes = {
    {0, 1, 2, 3}
};

const std::vector<std::vector<float>> defaultScales = {
    {1.f, 1.f, 2.f, 2.f}
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ngraph::op::v4::Interpolate::ShapeCalcMode::sizes,
        ngraph::op::v4::Interpolate::ShapeCalcMode::scales,
};

const auto interpolateCasesNearestMode2x = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesNearest2x),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));


const auto interpolateCasesNearestMode = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesNearest),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCasesNearestModeMore = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesNearestMore),
        ::testing::ValuesIn(defaultNearestModeMore),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCasesWithoutNearestMode = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_nearest_mode_no_batch, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearestMode,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapesNoBatch),
        ::testing::ValuesIn(targetShapesNoBatch),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_without_nearest_mode_no_batch, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestMode,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapesNoBatch),
        ::testing::ValuesIn(targetShapesNoBatch),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_nearest_mode, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearestMode,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_nearest_mode_more, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearestModeMore,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Interpolate_without_nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestMode,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(nightly_Interpolate_without_nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestMode,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapesNightly),
        ::testing::ValuesIn(targetShapesNightly),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(nightly_Interpolate_nearest_mode_2x, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearestMode2x,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes2x),
        ::testing::ValuesIn(targetShapes2x),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
        ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
    InterpolateLayerTest::getTestCaseName);

} // namespace
