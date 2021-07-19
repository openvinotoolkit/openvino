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

using groupConvBackpropDataLayerTestParamsSet = LayerTestsDefinitions::groupConvBackpropLayerTestParamsSet;
using groupConvBackpropDataSpecificParams = LayerTestsDefinitions::groupConvBackpropSpecificParams;

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
        result << LayerTestsDefinitions::GroupConvBackpropLayerTest::getTestCaseName(testing::TestParamInfo<groupConvBackpropDataLayerTestParamsSet>(
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
        groupConvBackpropDataLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        groupConvBackpropDataSpecificParams groupConvParams;
        std::vector<size_t> inputShape, outputShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(groupConvParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outputShape, targetDevice) = basicParamsSet;

        if (inPrc == Precision::UNSPECIFIED) {
            selectedType += std::string("_") + Precision(Precision::FP32).name();
        } else {
            selectedType += std::string("_") + inPrc.name();
        }

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd, outputPadding;
        size_t convOutChannels, numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType, outputPadding) = groupConvParams;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        std::shared_ptr<ngraph::op::v1::GroupConvolutionBackpropData> groupConv;
        if (!outputShape.empty()) {
            auto outShape = ngraph::opset3::Constant::create(ngraph::element::i64, {outputShape.size()}, outputShape);
            groupConv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(
            ngraph::builder::makeGroupConvolutionBackpropData(paramOuts[0], outShape, ngPrc, kernel, stride, padBegin,
                                                            padEnd, dilation, padType, convOutChannels, numGroups, false, outputPadding));
        } else {
            groupConv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(
            ngraph::builder::makeGroupConvolutionBackpropData(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                padEnd, dilation, padType, convOutChannels, numGroups, false, outputPadding));
        }
        function = makeNgraphFunction(ngPrc, params, groupConv, "groupConvolutionBackpropData");
    }
};

TEST_P(GroupDeconvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Deconvolution");
}

namespace {

/* GROUP CONV TEST UTILS */
std::vector<groupDeconvLayerCPUTestParamsSet> filterParamsSetForDevice(std::vector<groupDeconvLayerCPUTestParamsSet> paramsSet) {
    std::vector<groupDeconvLayerCPUTestParamsSet> resParamsSet;
    const int cpuParamsIndex = 1;
    const int selectedTypeIndex = 3;

    for (auto param : paramsSet) {
        auto cpuParams = std::get<cpuParamsIndex>(param);
        auto selectedTypeStr = std::get<selectedTypeIndex>(cpuParams);

        if (selectedTypeStr.find("jit") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !with_cpu_x86_avx512f())
            continue;

        resParamsSet.push_back(param);
    }

    return resParamsSet;
}
/* ===================== */

/* COMMON PARAMS */
std::vector<fusingSpecificParams> fusingParamsSet {
        emptyFusingSpec,
        fusingScaleShift,
};
const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES } };

const std::vector<std::vector<size_t >> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

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
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Planar_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Planar_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
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
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Planar_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Planar_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
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
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Blocked_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_Blocked_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
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
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Blocked_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_3D_Blocked_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 64, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
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
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_DW_FP32, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_DW_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 32, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupDeconv_2D_DW_BF16, GroupDeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            groupConvParams_ExplicitPadding_DW_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Precision::BF16),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 32, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    GroupDeconvolutionLayerCPUTest::getTestCaseName);
} // namespace

} // namespace CPULayerTestsDefinitions
