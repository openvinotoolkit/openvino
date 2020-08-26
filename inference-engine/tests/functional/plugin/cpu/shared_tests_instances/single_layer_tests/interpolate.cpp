// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> prc = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
        {1, 4, 30, 30},
};

const std::vector<std::vector<size_t>> targetShapes = {
        {1, 4, 40, 40},
};

const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modes = {
        ngraph::op::v4::Interpolate::InterpolateMode::linear,
        ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
        ngraph::op::v4::Interpolate::InterpolateMode::cubic,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
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

const std::vector<std::vector<size_t>> pads = {
        {0, 0, 1, 1},
        {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

/* ============= Mode is not equal to 'nearest' ============= */

const auto interpolateCasesNonNearest = ::testing::Combine(
        ::testing::ValuesIn(modes),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_CASE_P(Interpolate_with_not_nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNonNearest,
        ::testing::ValuesIn(prc),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    InterpolateLayerTest::getTestCaseName);

/* ============= Mode is equal to 'nearest' ============= */
const  std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearest_mode = {
        ngraph::op::v4::Interpolate::InterpolateMode::nearest,
};

const auto interpolateCases = ::testing::Combine(
        ::testing::ValuesIn(nearest_mode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs));

INSTANTIATE_TEST_CASE_P(Interpolate_Basic, InterpolateLayerTest, ::testing::Combine(
        interpolateCases,
        ::testing::ValuesIn(prc),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    InterpolateLayerTest::getTestCaseName);

} // namespace
