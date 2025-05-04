// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/interpolate.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test::Interpolate {
namespace {

// pillow modes: planar layout with axis[1,2] executed as nhwc layout case
const std::vector<int64_t> defaultAxes4D_pillow_nchw_as_nhwc = {
        {1, 2}
};

const std::vector<ShapeParams> shapeParams4D_Pillow_Smoke_nchw_as_nhwc = {
        ShapeParams{
                ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
                InputShape{{}, {{1, 4, 4, 3}}},
                ov::test::utils::InputLayerType::CONSTANT,
                {{2.0f, 4.0f}},
                defaultAxes4D_pillow_nchw_as_nhwc
        },
        ShapeParams{
                ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
                InputShape{{}, {{2, 16, 16, 4}}},
                ov::test::utils::InputLayerType::CONSTANT,
                {{2, 8}},
                defaultAxes4D_pillow_nchw_as_nhwc
        },
        ShapeParams{
                ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
                InputShape{{-1, -1, -1, {2, 20}}, {{1, 4, 4, 11}, {2, 6, 5, 7}, {1,  4, 4, 11}}},
                ov::test::utils::InputLayerType::CONSTANT,
                {{1.25f, 0.75f}},
                defaultAxes4D_pillow_nchw_as_nhwc
        },
        ShapeParams{
                ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
                InputShape{{-1, -1, -1, {2, 20}}, {{1, 4, 4, 17}, {2, 10, 12, 3}, {1, 4, 4, 17}}},
                ov::test::utils::InputLayerType::CONSTANT,
                {{6, 8}},
                defaultAxes4D_pillow_nchw_as_nhwc
        }
};

const std::vector<size_t> pads4D_nchw_as_nhwc = {
        {0, 0, 0, 0}
};

const std::vector<double> cubeCoefsPillow = {
        -0.5f,
};

// bilinear pillow and bicubic pillow test case supported in spec(ov ref)
const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModesPillow_Smoke = {
        ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
};

const auto interpolateCasesBilinearPillow_Smoke_nchw_as_nhwc = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::Values(pads4D_nchw_as_nhwc),
        ::testing::Values(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBilinearPillow_LayoutAlign_Test, InterpolateLayerCPUTest,
                         ::testing::Combine(
                                 interpolateCasesBilinearPillow_Smoke_nchw_as_nhwc,
                                 ::testing::ValuesIn(shapeParams4D_Pillow_Smoke_nchw_as_nhwc),
                                 ::testing::Values(ElementType::f32),
                                 ::testing::Values(CPUSpecificParams{{nchw, x, x}, {nchw}, {"ref"}, "ref"}),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(ov::AnyMap())),
                         InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesBicubicPillow_Smoke_nchw_as_nhwc = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::Values(pads4D_nchw_as_nhwc),
        ::testing::Values(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBicubicPillow_LayoutAlign_Test, InterpolateLayerCPUTest,
                         ::testing::Combine(
                                 interpolateCasesBicubicPillow_Smoke_nchw_as_nhwc,
                                 ::testing::ValuesIn(shapeParams4D_Pillow_Smoke_nchw_as_nhwc),
                                 ::testing::Values(ElementType::f32),
                                 ::testing::Values(CPUSpecificParams{{nchw, x, x}, {nchw}, {"ref"}, "ref"}),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(ov::AnyMap())),
                         InterpolateLayerCPUTest::getTestCaseName);

}  // namespace
} // namespace ov::test::Interpolate
