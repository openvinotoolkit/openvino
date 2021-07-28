// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/interpolate.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
            LayerTestsDefinitions::InterpolateLayerTestParams,
            CPUSpecificParams,
            fusingSpecificParams,
            std::map<std::string, std::string>
        > InterpolateLayerCPUTestParamsSet;

class InterpolateLayerCPUTest : public testing::WithParamInterface<InterpolateLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::InterpolateLayerTestParams>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }

protected:
    void SetUp() override {
        LayerTestsDefinitions::InterpolateLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        LayerTestsDefinitions::InterpolateSpecificParams interpolateParams;
        std::vector<size_t> inputShape;
        std::vector<size_t> targetShape;
        Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(interpolateParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape,
                 targetShape, targetDevice, additional_config) = basicParamsSet;

        ngraph::op::v4::Interpolate::InterpolateMode mode;
        ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
        ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
        ngraph::op::v4::Interpolate::NearestMode nearestMode;
        bool antialias;
        std::vector<size_t> padBegin, padEnd;
        double cubeCoef;
        std::vector<int64_t> axes;
        std::vector<float> scales;
        std::tie(mode, shapeCalcMode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef, axes, scales) = interpolateParams;
        inPrc = outPrc = netPrecision;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
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
        function = makeNgraphFunction(ngPrc, params, interpolate, "interpolate");
        if (selectedType.empty()) {
            selectedType = getPrimitiveType();
        }
        selectedType.push_back('_');
        if (additionalConfig.count(PluginConfigParams::KEY_ENFORCE_BF16) && additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            selectedType += "BF16";
        else
            selectedType += netPrecision.name();
    }
};

TEST_P(InterpolateLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Interpolate");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c, x, x, x}, {nChw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{nChw8c, x, x, x}, {nChw8c}, {"jit_sse42"}, "jit_sse42"});
        resCPUParams.push_back(CPUSpecificParams{{nhwc, x, x, x}, {nhwc}, {"jit_sse42"}, "jit_sse42"});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nchw, x, x, x}, {nchw}, {"ref"}, "ref"});
    }
    return resCPUParams;
}
/* ========== */

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
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
    {0, 1, 2, 3}
};

const std::vector<std::vector<float>> defaultScales = {
    {1.f, 1.f, 1.25f, 1.5f}
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

const auto interpolateCasesLinear = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear),
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

const std::vector<fusingSpecificParams> interpolateFusingParamsSet{
        emptyFusingSpec,
        fusingRelu,
        fusingSwish,
        fusingFakeQuantizePerChannelRelu,
};

std::map<std::string, std::string> additional_config = {};

std::vector<std::map<std::string, std::string>> filterAdditionalConfig() {
    if (with_cpu_x86_avx512f()) {
        return {
            {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::NO}},
            {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
        };
    } else {
        return {
            // default config as an stub for target without avx512, otherwise all tests with BF16 in its name are skipped
            {{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::NO}}
        };
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesNN,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 4, 4})),
                ::testing::Values(std::vector<size_t>({1, 21, 5, 6})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesLinearOnnx,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 4, 4})),
                ::testing::Values(std::vector<size_t>({1, 21, 5, 6})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinear_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesLinear,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 4, 4})),
                ::testing::Values(std::vector<size_t>({1, 21, 5, 6})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateCubic_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesCubic,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 4, 4})),
                ::testing::Values(std::vector<size_t>({1, 21, 5, 6})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
            ::testing::ValuesIn(filterCPUInfoForDevice()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

////////////////////////5D/////////////////////////////
std::vector<CPUSpecificParams> filterCPUInfoForDevice5D() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw16c, x, x, x}, {nCdhw16c}, {"jit_avx512"}, "jit_avx512"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx512"}, "jit_avx512"});
    } else if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c, x, x, x}, {nCdhw8c}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc, x, x, x}, {ndhwc}, {"jit_avx2"}, "jit_avx2"});
        resCPUParams.push_back(CPUSpecificParams{{ncdhw, x, x, x}, {ncdhw}, {"jit_avx2"}, "jit_avx2"});
    } else if (with_cpu_x86_sse42()) {
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

const std::vector<std::vector<float>> defaultScales5D = {
    {1.f, 1.f, 1.25f, 1.5f, 0.5f}
};

const auto interpolateCasesLinearOnnx5D = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes5D),
        ::testing::ValuesIn(defaultScales5D));

const auto interpolateCasesNN5D = ::testing::Combine(
        ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::nearest),
        ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModes),
        ::testing::ValuesIn(defNearestModes),
        ::testing::ValuesIn(antialias),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(pads5D),
        ::testing::ValuesIn(cubeCoefs),
        ::testing::ValuesIn(defaultAxes5D),
        ::testing::ValuesIn(defaultScales5D));

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateLinearOnnx5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesLinearOnnx5D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 4, 4, 4})),
                ::testing::Values(std::vector<size_t>({1, 21, 5, 6, 2})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_InterpolateNN5D_Layout_Test, InterpolateLayerCPUTest,
        ::testing::Combine(
            ::testing::Combine(
                interpolateCasesNN5D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 21, 4, 4, 4})),
                ::testing::Values(std::vector<size_t>({1, 21, 5, 6, 2})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
            ::testing::ValuesIn(filterCPUInfoForDevice5D()),
            ::testing::ValuesIn(interpolateFusingParamsSet),
            ::testing::ValuesIn(filterAdditionalConfig())),
    InterpolateLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
