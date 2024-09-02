// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/interpolate.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Interpolate {

const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModes_Full = {
    ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
    ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC,
    ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS
};

const std::vector<ov::op::v11::Interpolate::NearestMode> interpolateNearestModes = {
    ov::op::v11::Interpolate::NearestMode::SIMPLE,
    ov::op::v11::Interpolate::NearestMode::FLOOR
};

const std::vector<fusingSpecificParams> interpolateFusingParamsSet{
    emptyFusingSpec
};

const std::vector<ov::AnyMap> filterAdditionalConfig = {
    {ov::hint::inference_precision(ov::element::f32)},
    {ov::hint::inference_precision(ov::element::f16)}
};

const std::vector<CPUSpecificParams> filterCPUInfoForDevice = {
    CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"acl"}, "acl"},
    CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"acl"}, "acl"}
};

const std::vector<std::vector<size_t>> pads4D = {
    {0, 0, 0, 0}
};

const std::vector<std::vector<int64_t>> defaultAxes4D = {
    {0, 1, 2, 3}
};

const std::vector<ShapeParams> shapeParams4D_Smoke = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    }
};

const ShapeParams shapeParams4D_Scales = ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    };

const ShapeParams shapeParams4D_Sizes = ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {1, 11, 5, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    };

const std::vector<ShapeParams> shapeParams4D_Full = { shapeParams4D_Scales, shapeParams4D_Sizes };

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
             ::testing::Combine(
                 ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
                 ::testing::Values(ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC),
                 ::testing::ValuesIn(interpolateNearestModes),
                 ::testing::ValuesIn(antialias()),
                 ::testing::ValuesIn(pads4D),
                 ::testing::ValuesIn(pads4D),
                 ::testing::ValuesIn(cubeCoefs())),
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig)),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesNN_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::Values(ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS),
        ::testing::Values(ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_CEIL),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
         ::testing::Combine(
             interpolateCasesNN_Full,
             ::testing::ValuesIn(shapeParams4D_Full),
             ::testing::Values(ElementType::f32),
             ::testing::ValuesIn(filterCPUInfoForDevice),
             ::testing::ValuesIn(interpolateFusingParamsSet),
             ::testing::ValuesIn(filterAdditionalConfig)),
     InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinearOnnx_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Layout_Test_Sizes, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Full,
            ::testing::Values(shapeParams4D_Sizes),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig)),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinearOnnx_Scales = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::Values(ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Layout_Test_Scales, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Scales,
            ::testing::Values(shapeParams4D_Scales),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig)),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinear_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout_Test_sizes, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Full,
            ::testing::Values(shapeParams4D_Sizes),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig)),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinear_Scales = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::Values(ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout_Test_scales, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Scales,
            ::testing::Values(shapeParams4D_Scales),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig)),
    InterpolateLayerCPUTest::getTestCaseName);

// corner cases
const std::vector<ShapeParams> shapeParams4D_corner = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{1, 11, 4, 4}, {{1, 11, 4, 4}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{1, 11, 4, 4}, {{1, 11, 4, 4}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1, 11, 6, 7}, {1, 11, 8, 7}},
        defaultAxes4D.front()
    }
};

const auto interpolateCornerCases = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::Values(ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC),
        ::testing::Values(ov::op::v11::Interpolate::NearestMode::SIMPLE),
        ::testing::ValuesIn(antialias()),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::Values(std::vector<size_t>{0, 0, 0, 0}),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_corner_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCornerCases,
            ::testing::ValuesIn(shapeParams4D_corner),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig)),
    InterpolateLayerCPUTest::getTestCaseName);

}  // namespace Interpolate
}  // namespace test
}  // namespace ov
