// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::InterpolateLayerTest;
using ov::test::Interpolate11LayerTest;

class GPUInterpolateLayerTest : public InterpolateLayerTest {
protected:
    void SetUp() override {
        InterpolateLayerTest::SetUp();
        ov::test::InterpolateLayerTestParams params = GetParam();
        ov::test::InterpolateSpecificParams interpolate_params;
        ov::element::Type model_type;
        std::vector<ov::test::InputShape> shapes;
        ov::Shape target_shape;
        std::map<std::string, std::string> additional_config;
        std::tie(interpolate_params, model_type, shapes, target_shape, targetDevice, additional_config) = this->GetParam();
        // Some rounding float to integer types on GPU may differ from CPU, and as result,
        // the actual values may differ from reference ones on 1 when the float is very close to an integer,
        // e.g 6,0000023 calculated on CPU may be cast to 5 by OpenCL convert_uchar function.
        // That is why the threshold is set 1.f for integer types.
        if (targetDevice == "GPU" &&
                (model_type == ov::element::u8 || model_type == ov::element::i8)) {
            rel_threshold = 1.f;
        }
    }
};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f16,
        ov::element::f32,
};

const std::vector<ov::element::Type> netOnnx5dPrecisions = {
        ov::element::i8,
        ov::element::u8,
        ov::element::f16,
        ov::element::f32,
};

const std::vector<ov::Shape> inShapes = {
        {1, 1, 23, 23},
};

const std::vector<ov::Shape> targetShapes = {
        {1, 1, 46, 46},
};

const std::vector<ov::Shape> in5dShapes = {
        {1, 1, 2, 2, 2},
};

const std::vector<ov::Shape> target5dShapes = {
        {1, 1, 4, 4, 4},
};

const std::vector<ov::op::util::InterpolateBase::InterpolateMode> modesWithoutNearest = {
        ov::op::util::InterpolateBase::InterpolateMode::LINEAR,
        ov::op::util::InterpolateBase::InterpolateMode::CUBIC,
        ov::op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX,
};

const std::vector<ov::op::util::InterpolateBase::InterpolateMode> nearestMode = {
        ov::op::util::InterpolateBase::InterpolateMode::NEAREST,
};

const std::vector<ov::op::util::InterpolateBase::InterpolateMode> linearOnnxMode = {
        ov::op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX,
};

const std::vector<ov::op::util::InterpolateBase::CoordinateTransformMode> coordinateTransformModes = {
        ov::op::util::InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ov::op::util::InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ov::op::util::InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
        ov::op::util::InterpolateBase::CoordinateTransformMode::ASYMMETRIC,
        ov::op::util::InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ov::op::util::InterpolateBase::ShapeCalcMode> shapeCalculationMode = {
        ov::op::util::InterpolateBase::ShapeCalcMode::SIZES,
        ov::op::util::InterpolateBase::ShapeCalcMode::SCALES,
};

const std::vector<ov::op::util::InterpolateBase::NearestMode> nearestModes = {
        ov::op::util::InterpolateBase::NearestMode::SIMPLE,
        ov::op::util::InterpolateBase::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::util::InterpolateBase::NearestMode::FLOOR,
        ov::op::util::InterpolateBase::NearestMode::CEIL,
        ov::op::util::InterpolateBase::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ov::op::util::InterpolateBase::NearestMode> defaultNearestMode = {
        ov::op::util::InterpolateBase::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<std::vector<size_t>> pads = {
        {0, 0, 1, 1},
        {0, 0, 0, 0},
};

const std::vector<std::vector<size_t>> pads5dbegin = {
        {0, 0, 1, 1, 1},
        {0, 0, 0, 0, 0},
};
const std::vector<std::vector<size_t>> pads5dend = {
        {0, 0, 1, 1, 1},
        {0, 0, 0, 0, 0},
};

const std::vector<bool> antialias = {
// Not enabled in OpenVINO
//        true,
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxes = {
    {0, 1, 2, 3}
};

const std::vector<std::vector<int64_t>> emptyAxes = {{}};

const std::vector<std::vector<float>> defaultScales = {
    {1.f, 1.f, 2.f, 2.f}
};

const std::vector<std::vector<int64_t>> default5dAxes = {
    {0, 1, 2, 3, 4}
};

const std::vector<std::vector<float>> default5dScales = {
    {1.f, 1.f, 2.f, 2.f, 2.f}
};

std::map<std::string, std::string> additional_config = {};

const auto interpolateCasesWithoutNearest = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCasesWithoutNearestEmptyAxes = ::testing::Combine(
        ::testing::ValuesIn(modesWithoutNearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(emptyAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCasesNearesMode = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolate5dCasesLinearOnnxMode = ::testing::Combine(
        ::testing::ValuesIn(linearOnnxMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5dbegin),//pad begin
        ::testing::ValuesIn(pads5dend),//pad ends
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(default5dAxes),
        ::testing::ValuesIn(default5dScales));

const auto interpolate5dCasesNearestMode = ::testing::Combine(
        ::testing::ValuesIn(nearestMode),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5dbegin),//pad begin
        ::testing::ValuesIn(pads5dend),//pad ends
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(default5dAxes),
        ::testing::ValuesIn(default5dScales));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Basic, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(inShapes)),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_BasicEmptyAxes, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestEmptyAxes,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(inShapes)),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearesMode,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(inShapes)),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_5dLinearOnnx, GPUInterpolateLayerTest, ::testing::Combine(
        interpolate5dCasesLinearOnnxMode,
        ::testing::ValuesIn(netOnnx5dPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(in5dShapes)),
        ::testing::ValuesIn(target5dShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_5dNearest, GPUInterpolateLayerTest, ::testing::Combine(
        interpolate5dCasesNearestMode,
        ::testing::ValuesIn(netOnnx5dPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(in5dShapes)),
        ::testing::ValuesIn(target5dShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Basic, Interpolate11LayerTest, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(inShapes)),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);

const std::vector<ov::op::util::InterpolateBase::InterpolateMode> modesPillow = {
        ov::op::util::InterpolateBase::InterpolateMode::BILINEAR_PILLOW,
        ov::op::util::InterpolateBase::InterpolateMode::BICUBIC_PILLOW,
};

const std::vector<ov::element::Type> pillowModePrecisions = {
    ov::element::f16,
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ov::op::util::InterpolateBase::ShapeCalcMode::SCALES),
            ::testing::Values(ov::op::util::InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ov::op::util::InterpolateBase::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{2, 3}),
            ::testing::Values(std::vector<float>{2.f, 2.f})),
        ::testing::ValuesIn(pillowModePrecisions),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 23, 23}})),
        ::testing::Values(ov::Shape{1, 1, 50, 50}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow_Horizontal, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ov::op::util::InterpolateBase::ShapeCalcMode::SCALES),
            ::testing::Values(ov::op::util::InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ov::op::util::InterpolateBase::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{2, 3}),
            ::testing::Values(std::vector<float>{1.f, 2.f})),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 23, 23}})),
        ::testing::Values(ov::Shape{1, 1, 25, 50}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow_Vertical, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ov::op::util::InterpolateBase::ShapeCalcMode::SCALES),
            ::testing::Values(ov::op::util::InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ov::op::util::InterpolateBase::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{2, 3}),
            ::testing::Values(std::vector<float>{2.f, 1.f})),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 23, 23}})),
        ::testing::Values(ov::Shape{1, 1, 50, 25}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow_Vertical_BF, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ov::op::util::InterpolateBase::ShapeCalcMode::SCALES),
            ::testing::Values(ov::op::util::InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ov::op::util::InterpolateBase::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{2, 1, 0, 0}),
            ::testing::Values(std::vector<size_t>{2, 1, 0, 0}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{0, 1}),
            ::testing::Values(std::vector<float>{2.f, 1.f})),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{{23, 23, 2, 2}})),
        ::testing::Values(ov::Shape{52, 26, 2, 2}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);
} // namespace
