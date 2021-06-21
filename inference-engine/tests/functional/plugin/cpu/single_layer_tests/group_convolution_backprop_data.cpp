// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/group_convolution_backprop_data.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using groupConvBackpropDataLayerTestParamsSet = LayerTestsDefinitions::groupConvBackpropDataLayerTestParamsSet;
using groupConvBackpropDataSpecificParams = LayerTestsDefinitions::groupConvBackpropDataSpecificParams;

typedef std::tuple<
        groupConvBackpropDataLayerTestParamsSet,
        CPUSpecificParams,
        fusingSpecificParams,
        std::map<std::string, std::string>> groupDeconvLayerCPUTestParamsSet;

class GroupDeconvolutionLayerCPUTest : public testing::WithParamInterface<groupDeconvLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupDeconvLayerCPUTestParamsSet> obj) {
        groupConvBackpropDataLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::GroupConvBackpropDataLayerTest::getTestCaseName(testing::TestParamInfo<groupConvBackpropDataLayerTestParamsSet>(
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
    void SetUp() {
        using namespace ngraph;
        groupConvBackpropDataLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        groupConvBackpropDataSpecificParams groupConvParams;
        std::vector<size_t> inputShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(groupConvParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        if (inPrc == Precision::UNSPECIFIED)
            inPrc = Precision::FP32;
        if (outPrc == Precision::UNSPECIFIED)
            outPrc = Precision::FP32;

        if (inPrc == Precision::U8) {
            selectedType += std::string("_") + Precision(Precision::I8).name();
        } else {
            selectedType += std::string("_") + inPrc.name();
        }

        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;
        auto inElementType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto outElementType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
        if (inPrc == Precision::BF16)
            inElementType = element::f32;
        if (outPrc == Precision::BF16)
            outElementType = element::f32;

        auto inputParams = builder::makeParams(inElementType, { inputShape });
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        auto weiPrc = (inElementType == element::u8) ? element::i8 : inElementType;

        auto groupDeconvNode = ngraph::builder::makeGroupConvolutionBackpropDataRelaxed(paramOuts[0], weiPrc, outElementType, kernel,
                        stride, padBegin, padEnd, dilation, padType, convOutChannels, numGroups);
        function = makeNgraphFunction(element::f32, inputParams, groupDeconvNode, "groupConvolutionBackpropData");

        if (inPrc == Precision::U8 || inPrc == Precision::I8) {
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::i8, element::f32>>());
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::u8, element::f32>>());
        }
        if (outPrc != Precision::FP32 && outPrc != Precision::BF16) {
            additionalPasses.push_back(std::make_shared<ConvertPrecision<opset1::GroupConvolutionBackpropData>>());
        }
    }
};

TEST_P(GroupDeconvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Deconvolution");
}

namespace {

/* COMMON PARAMS */
std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
        fusingScaleShift,
};
const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES } };

/* ============= GroupConvolution params (planar layout) ============= */
const SizeVector numOutChannels_Planar = {6};
const SizeVector numGroups_Planar = {2, 3};

/* ============= GroupConvolution params (blocked layout) ============= */
const SizeVector numOutChannels_Blocked = {64};
const SizeVector numGroups_Blocked = {2, 4};

/* ============= GroupConvolution params (DW) ============= */
const SizeVector numOutChannels_DW = {32};
const SizeVector numGroups_DW = {32};

/* ============= GroupConvolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = {{3, 3}, {1, 1}};
const std::vector<SizeVector> strides2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2d = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2d = {{0, 0}};
const std::vector<SizeVector> dilations2d = {{1, 1}};

/* ============= GroupConvolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = {{3, 3, 3}, {1, 1, 1}};
const std::vector<SizeVector> strides3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3d = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3d = {{0, 0, 0}};
const std::vector<SizeVector> dilations3d = {{1, 1, 1}};
/* ============= */


/* INSTANCES */
/* ============= GroupConvolution (Planar 2D) ============= */
const auto groupConvParams_ExplicitPadding_Planar_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Planar),
        ::testing::ValuesIn(numGroups_Planar),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_Planar_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_Planar_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Planar 3D) ============= */
const auto groupConvParams_ExplicitPadding_Planar_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_Planar),
        ::testing::ValuesIn(numGroups_Planar),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_3D_Planar_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_3D_Planar_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 2D) ============= */
const auto groupConvParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_Blocked_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_Blocked_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 3D) ============= */
const auto groupConvParams_ExplicitPadding_Blocked_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_3D_Blocked_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_3D_Blocked_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (DW 2D) ============= */
const auto groupConvParams_ExplicitPadding_DW_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_DW),
        ::testing::ValuesIn(numGroups_DW),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_DW_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_DW_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 32, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_DW_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_DW_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 32, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution params I8 (2D) ============= */
const std::vector<fusingSpecificParams> fusingParamsSetI8{
        emptyFusingSpec,
        fusingRelu,
        fusingElu,
        fusingSigmoid,
        fusingClamp,
        fusingPReluPerChannel,
        fusingFakeQuantizePerChannel,
        fusingReluScaleShift,
};

const std::vector<SizeVector> kernels2di8 = { {3, 3} };
const std::vector<SizeVector> strides2di8 = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2di8 = { {0, 0}, {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds2di8 = { {0, 0}, {1, 1} };
const std::vector<SizeVector> dilations2di8 = { {1, 1} };

const auto groupDeconvParams_2D_I8 = ::testing::Combine(
        ::testing::ValuesIn(kernels2di8),
        ::testing::ValuesIn(strides2di8),
        ::testing::ValuesIn(padBegins2di8),
        ::testing::ValuesIn(padEnds2di8),
        ::testing::ValuesIn(dilations2di8),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_2D_I8 = {
        conv_sse42_2D_nspc,
        conv_avx2_2D_nspc,
        conv_avx512_2D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_I8, GroupDeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupDeconvParams_2D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution params 1x1 I8 (2D) ============= */
const auto groupDeconvParams_1x1_2D_I8 = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_2D_1x1_I8 = {
        // not supported 1x1 grouped conv for avx2/sse41 isa
        // conv_sse42_2D_1x1_I8,
        // conv_avx2_2D_1x1_I8,
        conv_avx512_2D_1x1_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_1x1_2D_I8, GroupDeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupDeconvParams_1x1_2D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_1x1_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution params I8 (3D) ============= */
const std::vector<SizeVector> kernels3di8 = { {3, 3, 3} };
const std::vector<SizeVector> strides3di8 = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3di8 = { {0, 0, 0}, {1, 1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds3di8 = { {0, 0, 0}, {1, 1, 1} };
const std::vector<SizeVector> dilations3di8 = { {1, 1, 1} };

const auto groupDeconvParams_3D_I8 = ::testing::Combine(
        ::testing::ValuesIn(kernels3di8),
        ::testing::ValuesIn(strides3di8),
        ::testing::ValuesIn(padBegins3di8),
        ::testing::ValuesIn(padEnds3di8),
        ::testing::ValuesIn(dilations3di8),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_3D_I8 = {
        conv_sse42_3D_nspc,
        conv_avx2_3D_nspc,
        conv_avx512_3D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_3D_I8, GroupDeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupDeconvParams_3D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution params 1x1 I8 (3D) ============= */
const auto groupDeconvParams_1x1_3D_I8 = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1, 1})),
        ::testing::Values(SizeVector({1, 1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(SizeVector({1, 1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_3D_1x1_I8 = {
        // not supported 1x1 grouped conv for avx2/sse41 isa
        // conv_sse42_3D_1x1_I8,
        // conv_avx2_3D_1x1_I8,
        conv_avx512_3D_1x1_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_1x1_3D_I8, GroupDeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupDeconvParams_1x1_3D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_1x1_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution params I8 (DW 2D) ============= */
const auto groupDeconvParams_DW_2D_I8 = ::testing::Combine(
        ::testing::ValuesIn(kernels2di8),
        ::testing::ValuesIn(strides2di8),
        ::testing::ValuesIn(padBegins2di8),
        ::testing::ValuesIn(padEnds2di8),
        ::testing::ValuesIn(dilations2di8),
        ::testing::ValuesIn(numOutChannels_DW),
        ::testing::ValuesIn(numGroups_DW),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_2D_DW_I8, GroupDeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupDeconvParams_DW_2D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8), // I8 and 3D DW deconvolution not supported in oneDNN
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 32, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        GroupDeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution params 1x1 I8 (DW 2D) ============= */
const auto groupDeconvParams_DW_1x1_2D_I8 = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::ValuesIn(numOutChannels_DW),
        ::testing::ValuesIn(numGroups_DW),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_GroupDeconv_DW_1x1_2D_I8, GroupDeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupDeconvParams_DW_1x1_2D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 32, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        GroupDeconvolutionLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
