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
        {40, 40},
};

const std::vector<std::set<size_t>> axes = {
        {2, 3},
};

const  std::vector<ngraph::op::v3::Interpolate::InterpolateMode> modes = {
        ngraph::op::v3::Interpolate::InterpolateMode::nearest,
        ngraph::op::v3::Interpolate::InterpolateMode::linear,
        ngraph::op::v3::Interpolate::InterpolateMode::linear_onnx,
        ngraph::op::v3::Interpolate::InterpolateMode::cubic,
        ngraph::op::v3::Interpolate::InterpolateMode::area,
};

const std::vector<ngraph::op::v3::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v3::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
        ngraph::op::v3::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v3::Interpolate::CoordinateTransformMode::half_pixel,
        ngraph::op::v3::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v3::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v3::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v3::Interpolate::NearestMode::simple,
        ngraph::op::v3::Interpolate::NearestMode::round_prefer_floor,
        ngraph::op::v3::Interpolate::NearestMode::floor,
        ngraph::op::v3::Interpolate::NearestMode::ceil,
        ngraph::op::v3::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<std::vector<size_t>> pads = {
        {1, 1},
        {0, 0},
};

const std::vector<bool> antialias = {
// Not enabled in Inference Engine
//        true,
        false,
};

const std::vector<double> cubeCoefs = {
        0.75f,
};

const auto interpolateCases = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(modes),
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