// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/interpolate.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::InterpolateLayerTestParams,
        CPUSpecificParams> InterpolateLayerCPUTestParamsSet;

class InterpolateLayerCPUTest : public testing::WithParamInterface<InterpolateLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::InterpolateLayerTestParams>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        LayerTestsDefinitions::InterpolateSpecificParams interpolateParams;
        std::vector<size_t> inputShape;
        std::vector<size_t> targetShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(interpolateParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShape, targetDevice) = basicParamsSet;

        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
        ngraph::op::v4::Interpolate::NearestMode nearestMode;
        bool antialias;
        std::vector<size_t> padBegin, padEnd;
        double cubeCoef;
        std::vector<int64_t> axes;
        std::vector<float> scales;
        std:tie(mode, shapeCalcMode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef, axes, scales) = interpolateParams;
        inPrc = outPrc = netPrecision;

        using ShapeCalcMode = ngraph::op::v4::Interpolate::ShapeCalcMode;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto constant = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {targetShape.size()}, targetShape);

        auto scales_const = ngraph::opset3::Constant(ngraph::element::Type_t::f32, {scales.size()}, scales);

        auto scalesInput = std::make_shared<ngraph::opset3::Constant>(scales_const);

        auto secondaryInput = std::make_shared<ngraph::opset3::Constant>(constant);

        auto axesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {axes.size()}, axes);
        auto axesInput = std::make_shared<ngraph::opset3::Constant>(axesConst);
        ngraph::op::v4::Interpolate::InterpolateAttrs interpolateAttributes{mode, shapeCalcMode, padBegin,
            padEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};
        auto interpolate = std::make_shared<ngraph::op::v4::Interpolate>(params[0],
                                                                         secondaryInput,
                                                                         scalesInput,
                                                                         axesInput,
                                                                         interpolateAttributes);
        interpolate->get_rt_info() = getCPUInfo();
        const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(interpolate)};
        function = std::make_shared<ngraph::Function>(results, params, "interpolate");

        selectedType = getPrimitiveType() + "_" + inPrc.name();
    }
};

TEST_P(InterpolateLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Interpolate");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512_FP32"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2_FP32"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42_FP32"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42_FP32"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x}, {nchw}, {"ref"}, "ref_FP32"});
    }
    return resCPUParams;
}
/* ========== */

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::BF16
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
        ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
        ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
        ngraph::op::v4::Interpolate::ShapeCalcMode::sizes,
        ngraph::op::v4::Interpolate::ShapeCalcMode::scales,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::simple,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
        ngraph::op::v4::Interpolate::NearestMode::floor,
        ngraph::op::v4::Interpolate::NearestMode::ceil,
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defNearestModes = {
        ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
};

const std::vector<std::vector<size_t>> pads = {
        {0, 0, 0, 0},
        {0, 0, 1, 1},
};

const std::vector<bool> antialias = {
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxes = {
    {2, 3}
};

const std::vector<std::vector<float>> defaultScales = {
    {1.25f, 1.5f}
};

const auto interpolateCasesNN = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCasesLinearOnnx = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

const auto interpolateCasesCubic = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::cubic),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes),
        ::testing::ValuesIn(defaultScales));

INSTANTIATE_TEST_CASE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesNN,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 40, 40})),
                ::testing::Values(std::vector<size_t>({1, 21, 50, 60})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesLinearOnnx,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 40, 40})),
                ::testing::Values(std::vector<size_t>({1, 21, 50, 60})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesCubic,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 40, 40})),
                ::testing::Values(std::vector<size_t>({1, 21, 50, 60})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            ::testing::ValuesIn(filterCPUInfoForDevice())),
    InterpolateLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
