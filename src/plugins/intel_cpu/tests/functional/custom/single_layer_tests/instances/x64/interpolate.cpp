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
namespace {

const std::vector<ov::AnyMap> filterAdditionalConfig() {
    if (ov::with_cpu_x86_avx512f()) {
        return {{{ov::hint::inference_precision(ov::element::f32)}},
                {{ov::hint::inference_precision(ov::element::bf16)}}};
    } else {
        return {// default config as an stub for target without avx512, otherwise all tests with BF16 in its name are
                // skipped
                {}};
    }
}

const std::vector<CPUSpecificParams> filterCPUInfoForDevice3D() {
    if (ov::with_cpu_x86_avx2()) {
        return { CPUSpecificParams{{ncw, x, x, x}, {ncw}, {"jit_avx2"}, "jit_avx2"} };
    } else {
        return { CPUSpecificParams{{ncw, x, x, x}, {ncw}, {"ref"}, "ref"} };
    }
}

const std::vector<ov::AnyMap> filterAdditionalConfig3D() {
    return { {} };
}

const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModes_Smoke = {
        ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModes_Full = {
        ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
        ov::op::v11::Interpolate::CoordinateTransformMode::ASYMMETRIC,
        ov::op::v11::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<ov::op::v11::Interpolate::NearestMode> nearestModes_Smoke = {
        ov::op::v11::Interpolate::NearestMode::SIMPLE,
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::v11::Interpolate::NearestMode::FLOOR,
};

const std::vector<ov::op::v11::Interpolate::NearestMode> nearestModes_Full = {
        ov::op::v11::Interpolate::NearestMode::SIMPLE,
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR,
        ov::op::v11::Interpolate::NearestMode::FLOOR,
        ov::op::v11::Interpolate::NearestMode::CEIL,
        ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<fusingSpecificParams> interpolateFusingParamsSet{
        emptyFusingSpec,
        fusingSwish,
        fusingFakeQuantizePerTensorRelu
};

const std::vector<std::vector<size_t>> pads3D_smoke = {
    {0, 0, 0},
};

const std::vector<std::vector<size_t>> pads3D_full = {
    {0, 0, 1},
};

const std::vector<std::vector<int64_t>> defaultAxes3D = {
    {0, 1, 2}
};

const std::vector<ShapeParams> shapeParams3D = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1}, {{1, 3, 4}, {2, 4, 6}, {1, 3, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f}, {1.f, 1.f, 1.25f}, {1.f, 1.f, 1.5f}},
        defaultAxes3D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1}, {{1, 3, 4}, {2, 4, 6}, {1, 3, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1, 3, 6}, {2, 4, 8}, {1, 3, 6}},
        defaultAxes3D.front()
    }
};

const auto interpolateCasesNN_Smoke_3D = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads3D_smoke),
        ::testing::ValuesIn(pads3D_smoke),
        ::testing::ValuesIn(cubeCoefs()));
INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test_3D, InterpolateLayerCPUTest,
        ::testing::Combine(
             interpolateCasesNN_Smoke_3D,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesNN_Full_3D = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads3D_full),
        ::testing::ValuesIn(pads3D_full),
        ::testing::ValuesIn(cubeCoefs()));
INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test_3D, InterpolateLayerCPUTest,
         ::testing::Combine(
             interpolateCasesNN_Full_3D,
             ::testing::ValuesIn(shapeParams3D),
             ::testing::Values(ElementType::f32),
             ::testing::ValuesIn(filterCPUInfoForDevice3D()),
             ::testing::ValuesIn(interpolateFusingParamsSet),
             ::testing::ValuesIn(filterAdditionalConfig3D())),
     InterpolateLayerCPUTest::getTestCaseName);

const std::vector<fusingSpecificParams> interpolateFusingParamsSet3D_fixed_C() {
    std::vector<fusingSpecificParams> fuseParams;
    if (ov::with_cpu_x86_avx2()) {
        fuseParams.push_back(fusingFakeQuantizePerChannelRelu);
        fuseParams.push_back(fusingMultiplyPerChannel);
    }
    fuseParams.push_back(emptyFusingSpec);
    return fuseParams;
}

const std::vector<ShapeParams> shapeParams3D_fixed_C = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 3, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f}},
        defaultAxes3D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, 3, -1}, {{1, 3, 4}, {1, 3, 6}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 3, 8}},
        defaultAxes3D.front()
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_PerChannelFuse3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Smoke_3D,
            ::testing::ValuesIn(shapeParams3D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet3D_fixed_C()),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_PerChannelFuse3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Full_3D,
            ::testing::ValuesIn(shapeParams3D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet3D_fixed_C()),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinear3D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads3D_smoke),
        ::testing::ValuesIn(pads3D_smoke),
        ::testing::ValuesIn(cubeCoefs()));

const auto interpolateCasesLinear3D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads3D_full),
        ::testing::ValuesIn(pads3D_full),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear3D_Smoke,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear3D_Full,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesCubic3D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads3D_smoke),
        ::testing::ValuesIn(pads3D_smoke),
        ::testing::ValuesIn(cubeCoefs()));

const auto interpolateCasesCubic3D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads3D_full),
        ::testing::ValuesIn(pads3D_full),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic3D_Smoke,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Layout3D_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic3D_Full,
            ::testing::ValuesIn(shapeParams3D),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice3D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig3D())),
    InterpolateLayerCPUTest::getTestCaseName);

// 4D
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (ov::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});
    } else if (ov::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

const std::vector<std::vector<size_t>> pads4D = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
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
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f}, {1.f, 1.f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1, 11, 6, 7}, {2, 7, 8, 7}, {1, 11, 6, 7}},
        defaultAxes4D.front()
    }
};

const std::vector<ShapeParams> shapeParams4D_Full = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {1, 11, 5, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6}},
        defaultAxes4D.front()
    }
};

const auto interpolateCasesNN_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

const auto interpolateCasesNN_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
             interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
         ::testing::Combine(
             interpolateCasesNN_Full,
             ::testing::ValuesIn(shapeParams4D_Full),
             ::testing::Values(ElementType::f32),
             ::testing::ValuesIn(filterCPUInfoForDevice()),
             ::testing::ValuesIn(interpolateFusingParamsSet),
             ::testing::ValuesIn(filterAdditionalConfig())),
     InterpolateLayerCPUTest::getTestCaseName);

const std::vector<fusingSpecificParams> interpolateFusingParamsSet_fixed_C{
        fusingFakeQuantizePerChannelRelu,
        fusingMultiplyPerChannel,
};

const std::vector<ShapeParams> shapeParams4D_fixed_C = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f}},
        defaultAxes4D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, 16, -1, -1}, {{1, 16, 4, 4}, {1, 16, 6, 5}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 16, 6, 7}},
        defaultAxes4D.front()
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Smoke,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet_fixed_C),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN_Layout_PerChannelFuse_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN_Full,
            ::testing::ValuesIn(shapeParams4D_fixed_C),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet_fixed_C),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinearOnnx_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

const auto interpolateCasesLinearOnnx_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesLinear_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

const auto interpolateCasesLinear_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinear_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinear_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesCubic_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

const auto interpolateCasesCubic_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::CUBIC),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Smoke,
            ::testing::ValuesIn(shapeParams4D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesCubic_Full,
            ::testing::ValuesIn(shapeParams4D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

////////////////////////5D/////////////////////////////
std::vector<CPUSpecificParams> filterCPUInfoForDevice5D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x, x, x}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (ov::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x, x, x}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"jit_avx2"}, "jit_avx2"});
    } else if (ov::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x, x, x}, {nCdhw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}

const std::vector<std::vector<size_t>> pads5D = {
        {0, 0, 0, 0, 0}
};

const std::vector<std::vector<int64_t>> defaultAxes5D = {
    {0, 1, 2, 3, 4}
};

const std::vector<ShapeParams> shapeParams5D_Smoke = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 2}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}, {1.f, 1.f, 1.25f, 1.25f, 1.25f}, {1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1, 11, 6, 7, 2}, {2, 7, 8, 7, 4}, {1, 11, 6, 7, 2}},
        defaultAxes5D.front()
    },
};

const std::vector<ShapeParams> shapeParams5D_Full = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {2, 7, 6, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.f, 1.f, 1.25f, 1.5f, 0.5f}},
        defaultAxes5D.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1, -1}, {{1, 11, 4, 4, 4}, {1, 11, 5, 5, 8}, {1, 11, 4, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1, 11, 5, 6, 4}},
        defaultAxes5D.front()
    }
};

const auto interpolateCasesLinearOnnx5D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs()));
const auto interpolateCasesLinearOnnx5D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesLinearOnnx5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesNN5D_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Smoke),
        ::testing::ValuesIn(nearestModes_Smoke),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs()));

const auto interpolateCasesNN5D_Full = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::NEAREST),
        ::testing::ValuesIn(coordinateTransformModes_Full),
        ::testing::ValuesIn(nearestModes_Full),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs()));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Smoke,
            ::testing::ValuesIn(shapeParams5D_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(InterpolateNN5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesNN5D_Full,
            ::testing::ValuesIn(shapeParams5D_Full),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
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
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

// bilinear pillow and bicubic pillow test case supported in spec(ov ref)
const std::vector<ov::op::v11::Interpolate::CoordinateTransformMode> coordinateTransformModesPillow_Smoke = {
    ov::op::v11::Interpolate::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
};

const std::vector<double> cubeCoefsPillow = {
    -0.5f,
};

const std::vector<fusingSpecificParams> interpolateFusingPillowParamsSet{
    emptyFusingSpec
};

const std::vector<std::vector<int64_t>> defaultAxes4D_pillow = {
    {2, 3}
};

const std::vector<ShapeParams> shapeParams4D_Pillow_Smoke = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 3, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{2.0f, 4.0f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{2, 4, 16, 16}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{0.25f, 0.5f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{1, 3, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{5, 6}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{2, 4, 16, 16}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{2, 8}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{1.25f, 1.5f}, {0.5f, 0.75f}, {1.25f, 1.5f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 11, 4, 4}, {2, 7, 6, 5}, {1, 11, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.25f, 0.75f}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 17, 4, 4}, {2, 3, 10, 12}, {1, 17, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{6, 8}, {5, 4}, {6, 8}},
        defaultAxes4D_pillow.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 17, 4, 4}, {2, 3, 10, 12}, {1, 17, 4, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{6, 8}},
        defaultAxes4D_pillow.front()
    },
    // test for only one pass or just copy
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, {2, 20}, -1, -1}, {{1, 17, 4, 4}, {2, 3, 10, 12}, {1, 17, 4, 4}}},
        ov::test::utils::InputLayerType::PARAMETER,
        {{4, 4}, {10, 20}, {10, 4}},
        defaultAxes4D_pillow.front()
    }
};

std::vector<CPUSpecificParams> filterCPUInfoForDevice_pillow() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (ov::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
    } else if (ov::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    }
    resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"ref"}, "ref"});
    return resCPUParams;
}
std::vector<ov::AnyMap> filterPillowAdditionalConfig() {
    return {{}};
}

const auto interpolateCasesBilinearPillow_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefsPillow));

const auto interpolateCasesBicubicPillow_Smoke = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(pads4D),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBilinearPillow_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBilinearPillow_Smoke,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBicubicPillow_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBicubicPillow_Smoke,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

// pillow modes: planar layout with axis[1,2] executed as nhwc layout case
const std::vector<std::vector<int64_t>> defaultAxes4D_pillow_nchw_as_nhwc = {
    {1, 2}
};

const std::vector<ShapeParams> shapeParams4D_Pillow_Smoke_nchw_as_nhwc = {
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{}, {{1, 4, 4, 3}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{2.0f, 4.0f}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{}, {{2, 16, 16, 4}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{2, 8}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SCALES,
        InputShape{{-1, -1, -1, {2, 20}}, {{1, 4, 4, 11}, {2, 6, 5, 7}, {1,  4, 4, 11}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{1.25f, 0.75f}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    },
    ShapeParams{
        ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
        InputShape{{-1, -1, -1, {2, 20}}, {{1, 4, 4, 17}, {2, 10, 12, 3}, {1, 4, 4, 17}}},
        ov::test::utils::InputLayerType::CONSTANT,
        {{6, 8}},
        defaultAxes4D_pillow_nchw_as_nhwc.front()
    }
};

const std::vector<std::vector<size_t>> pads4D_nchw_as_nhwc = {
        {0, 0, 0, 0}
};

std::vector<CPUSpecificParams> filterCPUInfoForDevice_pillow_nchw_as_nhwc() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_avx512"}, "jit_avx512"});
    } else if (ov::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});
    } else if (ov::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

const auto interpolateCasesBilinearPillow_Smoke_nchw_as_nhwc = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBilinearPillow_LayoutAlign_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBilinearPillow_Smoke_nchw_as_nhwc,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke_nchw_as_nhwc),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow_nchw_as_nhwc()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

const auto interpolateCasesBicubicPillow_Smoke_nchw_as_nhwc = ::testing::Combine(
        ::testing::Values(ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW),
        ::testing::ValuesIn(coordinateTransformModesPillow_Smoke),
        ::testing::ValuesIn(defNearestModes()),
        ::testing::ValuesIn(antialias()),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(pads4D_nchw_as_nhwc),
        ::testing::ValuesIn(cubeCoefsPillow));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateBicubicPillow_LayoutAlign_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            interpolateCasesBicubicPillow_Smoke_nchw_as_nhwc,
            ::testing::ValuesIn(shapeParams4D_Pillow_Smoke_nchw_as_nhwc),
            ::testing::Values(ElementType::f32),
            ::testing::ValuesIn(filterCPUInfoForDevice_pillow_nchw_as_nhwc()),
            ::testing::ValuesIn(interpolateFusingPillowParamsSet),
            ::testing::ValuesIn(filterPillowAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Interpolate
}  // namespace test
}  // namespace ov
