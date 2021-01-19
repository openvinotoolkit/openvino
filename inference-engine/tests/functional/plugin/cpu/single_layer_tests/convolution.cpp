// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <shared_test_classes/single_layer/convolution.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
using LayerTestsDefinitions::convSpecificParams;
using LayerTestsDefinitions::convLayerTestParamsSet;

typedef std::tuple<
    convLayerTestParamsSet,
    CPUSpecificParams,
    fusingSpecificParams,
    std::map<std::string, std::string> > convLayerCPUTestParamsSet;

class ConvolutionLayerCPUTest : public testing::WithParamInterface<convLayerCPUTestParamsSet>,
    virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convLayerCPUTestParamsSet> obj) {
        convLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<convLayerTestParamsSet>(
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
        convLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        convSpecificParams convParams;
        std::vector<size_t> inputShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        if (inPrc == Precision::UNSPECIFIED) {
            selectedType += std::string("_") + Precision(Precision::FP32).name();
        } else {
            selectedType += std::string("_") + inPrc.name();
        }

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto inputParams = ngraph::builder::makeParams(ngraph::element::f32, { inputShape });
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

        auto convolutionNode = ngraph::builder::makeConvolution(paramOuts.front(), ngPrc, kernel, stride, padBegin,
            padEnd, dilation, padType, convOutChannels);

        function = makeNgraphFunction(ngPrc, inputParams, convolutionNode, "Convolution");
    }
};

TEST_P(ConvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Convolution");
}

namespace {

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        // activations
        fusingRelu,
        fusingElu,
        fusingSigmoid,
        fusingClamp,
        fusingPRelu,
        fusingSwish,
        // other patterns
        fusingReluScaleShift,
        fusingFakeQuantizePerTensorRelu,
        fusingFakeQuantizePerChannelRelu,
        fusingSumEluFQ,
        fusingSum
};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{
        emptyFusingSpec,
        // activations
        fusingRelu,
        fusingElu,
        fusingSigmoid,
        fusingClamp,
        fusingPRelu,
        fusingSwish,
        // other patterns
        fusingReluScaleShift,
        fusingSum
};

const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES } };

/* ============= Convolution params (planar layout) ============= */
const SizeVector numOutChannels_Planar = { 6 };

/* ============= Convolution params (blocked layout) ============= */
const SizeVector numOutChannels_Blocked = { 64 };

/* ============= Convolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0}, {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1}, {2, 2} };

/* ============= Convolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0}, {1, 1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<SizeVector> dilations3d = { {1, 1, 1}, {2, 2, 2} };
/* ============= */

/* INSTANCES */
/* ============= Convolution (Planar 2D) ============= */
const auto convParams_ExplicitPadding_Planar_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_Planar_2D = {
        conv_gemm_2D
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_Planar_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_2D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_Planar_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_2D)),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Planar 3D) ============= */
const auto convParams_ExplicitPadding_Planar_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_Planar_3D = {
        conv_gemm_3D
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_Planar_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_3D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_Planar_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_3D)),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 2D) ============= */
const auto convParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_Blocked_2D = {
        conv_sse42_2D,
        conv_avx2_2D,
        conv_avx512_2D
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_Blocked_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_2D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_Blocked_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 3D) ============= */
const auto convParams_ExplicitPadding_Blocked_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_Blocked_3D = {
            //conv_sse42_3D, // not supported jit_sse42 for 3d
            conv_avx2_3D,
            conv_avx512_3D
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_Blocked_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_3D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_Blocked_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */

const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1x1_2D = {
    conv_sse42_2D_1x1,
    conv_avx2_2D_1x1,
    conv_avx512_2D_1x1
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_1x1_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_1x1_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_2D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_1x1_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_1x1_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1})),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ========= */

} // namespace
} // namespace CPULayerTestsDefinitions
