// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/interpolate.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

class GPUInterpolateLayerTest : public InterpolateLayerTest {
protected:
    void SetUp() override {
        InterpolateLayerTest::SetUp();
        InterpolateLayerTestParams params = GetParam();
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(std::ignore, netPrecision, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore, targetDevice, std::ignore) = params;
        // Some rounding float to integer types on GPU may differ from CPU, and as result,
        // the actual values may differ from reference ones on 1 when the float is very close to an integer,
        // e.g 6,0000023 calculated on CPU may be cast to 5 by OpenCL convert_uchar function.
        // That is why the threshold is set 1.f for integer types.
        if (targetDevice == "GPU" &&
                (netPrecision == InferenceEngine::Precision::U8 || netPrecision == InferenceEngine::Precision::I8)) {
            threshold = 1.f;
        }
    }
};

namespace v11 {

class GPUInterpolateLayerTest : public LayerTestsDefinitions::v11::InterpolateLayerTest {
protected:
    void SetUp() override {
        LayerTestsDefinitions::v11::InterpolateLayerTest::SetUp();
        InterpolateLayerTestParams params = GetParam();
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::tie(std::ignore, netPrecision, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore, targetDevice, std::ignore) = params;
        // Some rounding float to integer types on GPU may differ from CPU, and as result,
        // the actual values may differ from reference ones on 1 when the float is very close to an integer,
        // e.g 6,0000023 calculated on CPU may be cast to 5 by OpenCL convert_uchar function.
        // That is why the threshold is set 1.f for integer types.
        if (targetDevice == "GPU" &&
                (netPrecision == InferenceEngine::Precision::U8 || netPrecision == InferenceEngine::Precision::I8)) {
            threshold = 1.f;
        }
    }
};

} // namespace v11

TEST_P(GPUInterpolateLayerTest, CompareWithRefs) {
    Run();
}

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> netOnnx5dPrecisions = {
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
        {1, 1, 23, 23},
};

const std::vector<std::vector<size_t>> targetShapes = {
        {1, 1, 46, 46},
};

const std::vector<std::vector<size_t>> in5dShapes = {
        {1, 1, 2, 2, 2},
};

const std::vector<std::vector<size_t>> target5dShapes = {
        {1, 1, 4, 4, 4},
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR,
        ngraph::op::v4::Interpolate::InterpolateMode::CUBIC,
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> linearOnnxMode = {
        ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ngraph::op::v4::Interpolate::ShapeCalcMode::SIZES,
        ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::SIMPLE,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::FLOOR,
        ngraph::op::v4::Interpolate::NearestMode::CEIL,
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
        ngraph::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
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
// Not enabled in Inference Engine
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
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_BasicEmptyAxes, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesWithoutNearestEmptyAxes,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest, InterpolateLayerTest, ::testing::Combine(
        interpolateCasesNearesMode,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_5dLinearOnnx, GPUInterpolateLayerTest, ::testing::Combine(
        interpolate5dCasesLinearOnnxMode,
        ::testing::ValuesIn(netOnnx5dPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(in5dShapes),
        ::testing::ValuesIn(target5dShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_5dNearest, GPUInterpolateLayerTest, ::testing::Combine(
        interpolate5dCasesNearestMode,
        ::testing::ValuesIn(netOnnx5dPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(in5dShapes),
        ::testing::ValuesIn(target5dShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Basic, Interpolate11LayerTest, ::testing::Combine(
        interpolateCasesWithoutNearest,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(targetShapes),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesPillow = {
        ngraph::op::v4::Interpolate::InterpolateMode::BILINEAR_PILLOW,
        ngraph::op::v4::Interpolate::InterpolateMode::BICUBIC_PILLOW,
};

const std::vector<InferenceEngine::Precision> pillowModePrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES),
            ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{2, 3}),
            ::testing::Values(std::vector<float>{2.f, 2.f})),
        ::testing::ValuesIn(pillowModePrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 23, 23}),
        ::testing::Values(std::vector<size_t>{1, 1, 50, 50}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow_Horizontal, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES),
            ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{2, 3}),
            ::testing::Values(std::vector<float>{1.f, 2.f})),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 23, 23}),
        ::testing::Values(std::vector<size_t>{1, 1, 25, 50}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow_Vertical, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES),
            ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::Values(std::vector<size_t>{0, 0, 1, 1}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{2, 3}),
            ::testing::Values(std::vector<float>{2.f, 1.f})),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 1, 23, 23}),
        ::testing::Values(std::vector<size_t>{1, 1, 50, 25}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_11_Pillow_Vertical_BF, Interpolate11LayerTest, ::testing::Combine(
    ::testing::Combine(
            ::testing::ValuesIn(modesPillow),
            ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::SCALES),
            ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN),
            ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::SIMPLE),
            ::testing::Values(false),
            ::testing::Values(std::vector<size_t>{2, 1, 0, 0}),
            ::testing::Values(std::vector<size_t>{2, 1, 0, 0}),
            ::testing::ValuesIn(cubeCoefs),
            ::testing::Values(std::vector<int64_t>{0, 1}),
            ::testing::Values(std::vector<float>{2.f, 1.f})),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{23, 23, 2, 2}),
        ::testing::Values(std::vector<size_t>{52, 26, 2, 2}),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(additional_config)),
    Interpolate11LayerTest::getTestCaseName);
} // namespace
