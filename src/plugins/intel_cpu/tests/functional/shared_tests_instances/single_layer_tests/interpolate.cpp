// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::InterpolateLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<ov::Shape> input_shapes_static = {
    {1, 4, 6, 6}
};

const  std::vector<ov::op::v4::Interpolate::InterpolateMode> modes_without_nearest = {
    ov::op::v4::Interpolate::InterpolateMode::LINEAR,
    ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
    ov::op::v4::Interpolate::InterpolateMode::CUBIC,
};

const  std::vector<ov::op::v4::Interpolate::InterpolateMode> nearest_mode = {
    ov::op::v4::Interpolate::InterpolateMode::NEAREST,
};

const std::vector<ov::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
    ov::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
    ov::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
    ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
    ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
    ov::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ov::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
    ov::op::v4::Interpolate::ShapeCalcMode::SIZES,
    ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
};

const std::vector<ov::op::v4::Interpolate::NearestMode> nearest_modes = {
    ov::op::v4::Interpolate::NearestMode::SIMPLE,
    ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
    ov::op::v4::Interpolate::NearestMode::FLOOR,
    ov::op::v4::Interpolate::NearestMode::CEIL,
    ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ov::op::v4::Interpolate::NearestMode> default_nearest_mode = {
    ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<std::vector<size_t>> pads = {
    {0, 0, 1, 1},
    {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
// Not enabled in OpenVINO
//        true,
    false,
};

const std::vector<double> cube_coefs = {
    -0.75f,
};

const std::vector<std::vector<int64_t>> default_axes = {
    {0, 1, 2, 3}
};

const std::vector<ov::Shape> target_shapes = {
    {1, 4, 8, 8},
};

const std::vector<std::vector<float>> default_scales = {
    {1.f, 1.f, 1.333333f, 1.333333f}
};

std::map<std::string, std::string> additional_config = {};

const auto interpolateCasesWithoutNearest = ::testing::Combine(
        ::testing::ValuesIn(modes_without_nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(default_nearest_mode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cube_coefs),
        ::testing::ValuesIn(default_axes),
        ::testing::ValuesIn(default_scales));

const auto interpolateCases = ::testing::Combine(
        ::testing::ValuesIn(nearest_mode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearest_modes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cube_coefs),
        ::testing::ValuesIn(default_axes),
        ::testing::ValuesIn(default_scales));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Basic, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
        ::testing::ValuesIn(target_shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCases,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
        ::testing::ValuesIn(target_shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

const std::vector<ov::Shape> target_shapes_tail_test = {
    {1, 4, 2, 11},  // cover down sample and tails process code path
};

const std::vector<std::vector<float>> default_scalesTailTest = {
    {1.f, 1.f, 0.333333f, 1.833333f}
};

const auto interpolateCasesWithoutNearestTail = ::testing::Combine(
        ::testing::ValuesIn(modes_without_nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(default_nearest_mode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cube_coefs),
        ::testing::ValuesIn(default_axes),
        ::testing::ValuesIn(default_scalesTailTest));

const auto interpolateCasesTail = ::testing::Combine(
        ::testing::ValuesIn(nearest_mode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearest_modes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cube_coefs),
        ::testing::ValuesIn(default_axes),
        ::testing::ValuesIn(default_scalesTailTest));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Basic_Down_Sample_Tail, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestTail,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
        ::testing::ValuesIn(target_shapes_tail_test),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Down_Sample_Tail, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesTail,
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::test::static_shapes_to_test_representation(input_shapes_static)),
        ::testing::ValuesIn(target_shapes_tail_test),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

} // namespace
