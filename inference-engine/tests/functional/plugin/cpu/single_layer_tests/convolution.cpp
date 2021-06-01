// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
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
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerCPUTestParamsSet>& obj) {
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
    bool isBias = false;

    void checkBiasFusing(InferenceEngine::ExecutableNetwork &execNet) const {
        auto execGraph = execNet.GetExecGraphInfo().getFunction();
        ASSERT_NE(nullptr, execGraph);

        bool foundConv = false;
        for (const auto &node : execGraph->get_ops()) {
            const auto & rtInfo = node->get_rt_info();
            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                IE_ASSERT(rtInfo.end() != it);
                auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
                IE_ASSERT(nullptr != value);
                return value->get();
            };

            if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Convolution") {
                foundConv = true;
                ASSERT_EQ(3, node->inputs().size());
                break;
            }
        }

        ASSERT_TRUE(foundConv) << "Can't find Convolution node";
    }

    void SetUp() override {
        convLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        if (postOpMgrPtr)
            isBias = (postOpMgrPtr->getFusedOpsNames() == "Add(PerChannel)" && selectedType != "jit_avx512_winograd");

        convSpecificParams convParams;
        std::vector<size_t> inputShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        if (inPrc == Precision::UNSPECIFIED) {
            selectedType += std::string("_") + Precision(Precision::FP32).name();
        } else if (inPrc == Precision::BF16) {
            selectedType += std::string("_") + inPrc.name();
        } else {
            selectedType += std::string("_") + Precision(netPrecision).name();
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

    // Skip tests for sse41 convolution where ic or oc cannot be exactly divided by the block size,
    // since tails processing for sse41 nspc layout is not supported yet (see 52736).
    if (!inFmts.empty() && (inFmts.front() == nhwc || inFmts.front() == ndhwc) && selectedType.find("jit_sse") != std::string::npos) {
        auto inpChannels = function->get_parameters().front()->get_shape()[1];
        auto outChannels = function->get_output_shape(0)[1];
        if ((inpChannels % 8) || (outChannels % 8)) {
            GTEST_SKIP() << "Disabled test due to the sse41 convolution kernel does not support tails for nspc layout." << std::endl;
        }
    }

    Run();

    if (isBias) {
        checkBiasFusing(executableNetwork);
    }
    CheckPluginRelatedResults(executableNetwork, "Convolution");
}

namespace {

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        // eltwise
        fusingRelu,
        fusingPRelu1D,
        // depthwise
        fusingReluScaleShift,
        // fake quantize
        fusingFakeQuantizePerTensorRelu,
        fusingFakeQuantizePerChannelRelu,
        // sum
        fusingSumEluFQ,
        fusingSum,
        // bias
        fusingAddPerChannel
};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{
        emptyFusingSpec,
        // eltwise
        fusingRelu,
        // depthwise
        fusingReluScaleShift,
        // sum
        fusingSum,
        // bias
        fusingAddPerChannel
};

const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES } };

/* ============= Convolution params (GEMM layout) ============= */
const SizeVector numOutChannels_Gemm = {6 };

/* ============= Convolution params (blocked and nspc layout) ============= */
const SizeVector numOutChannels = { 64, 63 };

/* ============= Convolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0}, {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1}, {2, 2} };
const std::vector<SizeVector> inputShapes2d = { {1, 64, 7, 7}, {1, 67, 7, 7} };
const std::vector<SizeVector> inputShapesPlain2Blocked2d = { {1, 1, 7, 7}, {1, 2, 7, 7},  {1, 3, 7, 7} };

/* ============= Convolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0}, {1, 1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<SizeVector> dilations3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<SizeVector> inputShapes3d = { {1, 64, 7, 7, 7}, {1, 67, 7, 7, 7} };
const std::vector<SizeVector> inputShapesPlain2Blocked3d = { {1, 1, 7, 7, 7}, {1, 2, 7, 7, 7},  {1, 3, 7, 7, 7} };
/* ============= */

/* INSTANCES */
/* ============= Convolution (Gemm 2D) ============= */
const auto convParams_ExplicitPadding_GEMM_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Gemm),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_2D = {
        conv_gemm_2D,
        conv_gemm_2D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_GEMM_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_GEMM_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_GEMM_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_GEMM_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_GEMM_I8, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_GEMM_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::I8),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
        ::testing::Values(fusingSum),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (GEMM 3D) ============= */
const auto convParams_ExplicitPadding_GEMM_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Gemm),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_3D = {
        conv_gemm_3D,
        conv_gemm_3D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_GEMM_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_GEMM_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_GEMM_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_GEMM_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_GEMM_I8, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_GEMM_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::I8),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
        ::testing::Values(fusingSum),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (2D) ============= */
const auto convParams_ExplicitPadding_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_2D = {
        conv_sse42_2D,
        conv_avx2_2D,
        conv_avx512_2D,
        conv_sse42_2D_nspc,
        conv_avx2_2D_nspc,
        conv_avx512_2D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D, conv_avx512_2D_nspc})),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_I8, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::I8),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
        ::testing::Values(fusingSum),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_2D_plain_to_blocked = {
        conv_sse42_plain_to_blocked_2D,
        conv_avx2_plain_to_blocked_2D,
        conv_avx512_plain_to_blocked_2D,
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_PlainToBlocked_2D_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapesPlain2Blocked2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_plain_to_blocked)),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_PlainToBlocked_2D_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16, Precision::FP32),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapesPlain2Blocked2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_2D})),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (3D) ============= */
const auto convParams_ExplicitPadding_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_3D = {
            //conv_sse42_3D, // not supported jit_sse42 for 3d
            conv_avx2_3D,
            conv_avx512_3D,
            conv_avx2_3D_nspc,
            conv_avx512_3D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes3d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes3d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D, conv_avx512_3D_nspc})),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_3D_I8, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::I8),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes3d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
        ::testing::Values(fusingSum),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_3D_plain_to_blocked = {
        conv_avx2_plain_to_blocked_3D,
        conv_avx512_plain_to_blocked_3D,
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_PlainToBlocked_3D_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapesPlain2Blocked3d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_plain_to_blocked)),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_PlainToBlocked_3D_BF16, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16, Precision::FP32),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapesPlain2Blocked3d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_3D})),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */

const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(63),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1x1_2D = {
    conv_sse42_2D_1x1,
    conv_avx2_2D_1x1,
    conv_avx512_2D_1x1,
    conv_sse42_2D_1x1_nspc,
    conv_avx2_2D_1x1_nspc,
    conv_avx512_2D_1x1_nspc
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
            ::testing::ValuesIn(inputShapes2d),
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
            ::testing::ValuesIn(inputShapes2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1, conv_avx512_2D_1x1_nspc})),
        ::testing::ValuesIn(fusingParamsSetBF16),
        ::testing::Values(cpuBF16PluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Conv_2D_1x1_I8, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_1x1_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::I8),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_2D)),
        ::testing::Values(fusingSum),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (1D) ============= */
/* ============= Convolution params (1D) ============= */
const std::vector<SizeVector> kernels1d = { {3} };
const std::vector<SizeVector> strides1d = { {1}, {2} };
const std::vector<std::vector<ptrdiff_t>> padBegins1d = { {0}, {1} };
const std::vector<std::vector<ptrdiff_t>> padEnds1d = { {0} };
const std::vector<SizeVector> dilations1d = { {1}, {2} };

const auto convParams_1D = ::testing::Combine(
    ::testing::ValuesIn(kernels1d),
    ::testing::ValuesIn(strides1d),
    ::testing::ValuesIn(padBegins1d),
    ::testing::ValuesIn(padEnds1d),
    ::testing::ValuesIn(dilations1d),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1D = {
        conv_sse42_1D,
        conv_avx2_1D,
        conv_avx512_1D
};

INSTANTIATE_TEST_CASE_P(smoke_Conv_1D, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_1D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D)),
        ::testing::Values(fusingAddPerChannel),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Jit Planar ============= */

/* ============= Convolution planar params (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_2D = {
        // sse42 is not supported
        conv_avx2_planar_2D,
        conv_avx512_planar_2D,
};

const auto convParams_Planar_ExplicitPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Conv_Jit_Planar_2D_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_Planar_ExplicitPadding_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes2d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_2D)),
        ::testing::Values(emptyFusingSpec, fusingRelu),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution planar params (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_3D = {
        // sse42 is not supported
        conv_avx2_planar_3D,
        conv_avx512_planar_3D,
};

const auto convParams_Planar_ExplicitPadding_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::Values(SizeVector{1, 1, 1}),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Conv_Jit_Planar_3D_FP32, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_Planar_ExplicitPadding_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::ValuesIn(inputShapes3d),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_3D)),
        ::testing::Values(emptyFusingSpec, fusingRelu),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

/* ============= */

} // namespace


/* ============= Winograd ============= */
namespace winograd {

const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        fusingRelu,
        fusingSum,
        fusingAddPerChannel // bias
};

const SizeVector numOutChannels = { 32 };

const std::vector<SizeVector> kernels2d = { {3, 3} };
const std::vector<SizeVector> strides2d = { {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1} };

const auto convParams_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Conv_winograd, ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 1, 16, 10, 10 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice(std::vector<CPUSpecificParams>{conv_winograd})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    ConvolutionLayerCPUTest::getTestCaseName);

} // namespace winograd

} // namespace CPULayerTestsDefinitions
