// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/group_convolution.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using groupConvLayerTestParamsSet = LayerTestsDefinitions::groupConvLayerTestParamsSet;
using groupConvSpecificParams = LayerTestsDefinitions::groupConvSpecificParams;

typedef std::tuple<
        groupConvLayerTestParamsSet,
        CPUSpecificParams,
        fusingSpecificParams> groupConvLayerCPUTestParamsSet;

class GroupConvolutionLayerCPUTest : public testing::WithParamInterface<groupConvLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvLayerCPUTestParamsSet> obj) {
        groupConvLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::GroupConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<groupConvLayerTestParamsSet>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

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

    void SetUp() {
        groupConvLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        if (postOpMgrPtr)
                isBias = postOpMgrPtr->getFusedOpsNames() == "Add(PerChannel)";

        groupConvSpecificParams groupConvParams;
        std::vector<size_t> inputShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(groupConvParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

        if (inPrc == Precision::UNSPECIFIED) {
            selectedType += std::string("_") + Precision(Precision::FP32).name();
        } else {
            selectedType += std::string("_") + inPrc.name();
        }

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto groupConv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(
                ngraph::builder::makeGroupConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                      padEnd, dilation, padType, convOutChannels, numGroups));
        function = makeNgraphFunction(ngPrc, params, groupConv, "groupConvolution");
    }
};

TEST_P(GroupConvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    if (isBias) {
        checkBiasFusing(executableNetwork);
    }
    CheckPluginRelatedResults(executableNetwork, "Convolution");
}

namespace {

/* GROUP CONV TEST UTILS */
std::vector<groupConvLayerCPUTestParamsSet> filterParamsSetForDevice(std::vector<groupConvLayerCPUTestParamsSet> paramsSet) {
    std::vector<groupConvLayerCPUTestParamsSet> resParamsSet;
    const int cpuParamsIndex = 1;
    const int selectedTypeIndex = 3;

    for (auto param : paramsSet) {
        auto cpuParams = std::get<cpuParamsIndex>(param);
        auto selectedTypeStr = std::get<selectedTypeIndex>(cpuParams);

        if (selectedTypeStr.find("jit") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !with_cpu_x86_avx512f())
            continue;

        resParamsSet.push_back(param);
    }

    return resParamsSet;
}
/* ===================== */

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet {
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
        fusingSum
};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{
        emptyFusingSpec,
        // eltwise
        fusingRelu,
        // depthwise
        fusingReluScaleShift,
        // sum
        fusingSum
};


/* ============= GroupConvolution params (planar layout) ============= */
const SizeVector numOutChannels_Gemm = {6};
const SizeVector numGroups_Gemm = {2, 3};

/* ============= GroupConvolution params (blocked layout) ============= */
const SizeVector numOutChannels_Blocked = {64};
const SizeVector numGroups_Blocked = {2, 4};

/* ============= GroupConvolution params (DW) ============= */
const SizeVector numOutChannels_DW = {32};
const SizeVector numGroups_DW = {32};

/* ============= GroupConvolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = {{3, 3}, {1, 1}};
const std::vector<SizeVector> strides2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2d = {{0, 0}, {1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds2d = {{0, 0}};
const std::vector<SizeVector> dilations2d = {{1, 1}, {2, 2}};

/* ============= GroupConvolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = {{3, 3, 3}, {1, 1, 1}};
const std::vector<SizeVector> strides3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3d = {{0, 0, 0}, {1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds3d = {{0, 0, 0}};
const std::vector<SizeVector> dilations3d = {{1, 1, 1}, {2, 2, 2}};
/* ============= */


/* INSTANCES */
/* ============= GroupConvolution (GEMM 2D) ============= */
const auto groupConvParams_ExplicitPadding_Gemm_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::ValuesIn(numGroups_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_Gemm_2D = {
        conv_gemm_2D,
        conv_gemm_2D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_2D_Gemm_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Gemm_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 12, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_2D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_2D_Gemm_BF16, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Gemm_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 12, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_2D)),
                                ::testing::ValuesIn(fusingParamsSetBF16)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (Gemm 3D) ============= */
const auto groupConvParams_ExplicitPadding_Gemm_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::ValuesIn(numGroups_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_Gemm_3D = {
        conv_gemm_3D,
        conv_gemm_3D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_3D_Gemm_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Gemm_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 12, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_3D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_3D_Gemm_BF16, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Gemm_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 12, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Gemm_3D)),
                                ::testing::ValuesIn(fusingParamsSetBF16)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (2D) ============= */
const auto groupConvParams_ExplicitPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
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

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_2D_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 64, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_2D_BF16, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 64, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D, conv_avx512_2D_nspc})),
                                ::testing::ValuesIn(fusingParamsSetBF16)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (3D) ============= */
const auto groupConvParams_ExplicitPadding_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::ValuesIn(numGroups_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_3D = {
//        conv_sse42_3D, // not supported jit_sse42 for 3d
        conv_avx2_3D,
        conv_avx512_3D,
        conv_avx2_3D_nspc,
        conv_avx512_3D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_3D_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 64, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_3D_BF16, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 64, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D, conv_avx512_3D_nspc})),
                                ::testing::ValuesIn(fusingParamsSetBF16)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

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

const std::vector<CPUSpecificParams> CPUParams_DW_2D = {
        conv_sse42_dw_2D,
        conv_avx2_dw_2D,
        conv_avx512_dw_2D,
        conv_sse42_dw_2D_nspc,
        conv_avx2_dw_2D_nspc,
        conv_avx512_dw_2D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_2D_DW_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 32, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_2D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);


INSTANTIATE_TEST_CASE_P(smoke_GroupConv_2D_DW_BF16, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(Precision::BF16),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 32, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_dw_2D, conv_avx512_dw_2D_nspc})),
                                ::testing::ValuesIn(fusingParamsSetBF16)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupConvolution (DW 3D) ============= */
const auto groupConvParams_ExplicitPadding_DW_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_DW),
        ::testing::ValuesIn(numGroups_DW),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_DW_3D = {
        conv_sse42_dw_3D,
        conv_avx2_dw_3D,
        conv_avx512_dw_3D,
        conv_sse42_dw_3D_nspc,
        conv_avx2_dw_3D_nspc,
        conv_avx512_dw_3D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_3D_DW_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(InferenceEngine::Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({2, 32, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_3D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);
/* ========= */


/* ============= SINGLE TEST CASES ============= */
using VecFusingParams = std::vector<fusingSpecificParams>;
using PrcConnectedParams = std::tuple<Precision, Precision,  VecFusingParams>; // inPrc, outPrc, FusingParamsSet
using VecPrcConnectedParams = std::vector<PrcConnectedParams>;

std::vector<groupConvLayerCPUTestParamsSet> makeSingleGroupConvCPUTestCases(SizeVector kernels, SizeVector strides, SizeVector dilations,
                                                                            std::vector<ptrdiff_t> padBegins, std::vector<ptrdiff_t> padEnds,
                                                                            ngraph::op::PadType padType, int groups, int mb, SizeVector spDims,
                                                                            int inGroupSize, int outGroupSize,
                                                                            const std::vector<CPUSpecificParams>& CPUParams,
                                                                            const VecPrcConnectedParams& vecPrcConnectedParams) {
    int inChannels = groups * inGroupSize;
    int outChannels = groups * outGroupSize;

    SizeVector inputShapes;
    inputShapes.push_back(mb);
    inputShapes.push_back(inChannels);
    inputShapes.insert(inputShapes.end(), spDims.begin(), spDims.end());

    groupConvSpecificParams specificParams(kernels, strides, padBegins, padEnds, dilations, outChannels, groups, padType);
    std::vector<groupConvLayerCPUTestParamsSet> retVector;

    for (auto& prcConnectedParams : vecPrcConnectedParams) {
        Precision inPrc, outPrc;
        VecFusingParams fusingParams;
        std::tie(inPrc, outPrc, fusingParams) = prcConnectedParams;

        groupConvLayerTestParamsSet basicParamsSet(specificParams, Precision::FP32, inPrc, outPrc,
                                                   InferenceEngine::Layout::ANY, InferenceEngine::Layout::ANY,
                                                   inputShapes, CommonTestUtils::DEVICE_CPU);

        for (auto &item : CPUParams) {
            for (auto &fusingParam : fusingParams) {
                retVector.push_back(groupConvLayerCPUTestParamsSet(basicParamsSet, item, fusingParam));
            }
        }
    }
    return  retVector;
}

template<typename T>
void concatTestCases(std::vector<groupConvLayerCPUTestParamsSet>& resultVec, T tesCase) {
    resultVec.insert(resultVec.begin(), std::make_move_iterator(tesCase.begin()), std::make_move_iterator(tesCase.end()));
}

template<typename T, typename... Args>
void concatTestCases(std::vector<groupConvLayerCPUTestParamsSet>& resultVec, T&& tesCase, Args&&... args) {
    concatTestCases(resultVec, std::forward<T>(tesCase));
    concatTestCases(resultVec, std::forward<Args>(args)...);
}

template<typename... Args>
std::vector<groupConvLayerCPUTestParamsSet> generateSingleGroupConvCPUTestCases(Args&&... args) {
    std::vector<groupConvLayerCPUTestParamsSet> retVec;
    concatTestCases(retVec, std::forward<Args>(args)...);
    return retVec;
}

/* COMMON PARAMS */

const VecPrcConnectedParams vecPrcConnectParamsFP32 = {PrcConnectedParams{Precision::FP32, Precision::FP32, fusingParamsSet}};
const VecPrcConnectedParams vecPrcConnectParams = {PrcConnectedParams{Precision::FP32, Precision::FP32, fusingParamsSet},
                                                   PrcConnectedParams{Precision::BF16, Precision::BF16, fusingParamsSetBF16},
                                                   PrcConnectedParams{Precision::BF16, Precision::FP32, fusingParamsSetBF16}};

const VecPrcConnectedParams vecPrcConnectParamsFP32Default = {PrcConnectedParams{Precision::FP32, Precision::FP32, VecFusingParams{emptyFusingSpec}}};
const VecPrcConnectedParams vecPrcConnectParamsDefault = {PrcConnectedParams{Precision::FP32, Precision::FP32, VecFusingParams{emptyFusingSpec}},
                                                          PrcConnectedParams{Precision::BF16, Precision::BF16, VecFusingParams{emptyFusingSpec}},
                                                          PrcConnectedParams{Precision::BF16, Precision::FP32, VecFusingParams{emptyFusingSpec}}};

/* ============= GEMM GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> gemmGroupConvTestCases = generateSingleGroupConvCPUTestCases(
        //  1. is_depthwise (true, false)
        //  2. jcp.im2col_sz (=0,>0)
        //  3. is_blocking_applicable (true, false)

        //  is_depthwise == false, im2col_sz > 0
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 2, 2, CPUParams_Gemm_2D, vecPrcConnectParams),
        //  is_depthwise == true
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 1, 1,
                                        CPUParams_Gemm_2D, vecPrcConnectParams),
        //  im2col_sz == 0, is_blocking_applicable == true
        makeSingleGroupConvCPUTestCases({1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 2, 2, CPUParams_Gemm_2D, vecPrcConnectParams),
        //  is_blocking_applicable == false ((jcp.im2col_sz == 0) && (jcp.ic / jcp.oc >= 42))
        makeSingleGroupConvCPUTestCases({1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 42, 1, CPUParams_Gemm_2D, vecPrcConnectParams),

        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT,
                                        3, 2, {129, 129}, 4, 2, CPUParams_Gemm_2D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT,
                                        2, 1, {10, 10}, 3, 3, CPUParams_Gemm_2D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                        3, 2, {33, 33, 33}, 4, 2, CPUParams_Gemm_3D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                        2, 1, {10, 10, 10}, 3, 3, CPUParams_Gemm_3D, vecPrcConnectParams)
);

INSTANTIATE_TEST_CASE_P(smoke_GEMM_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(gemmGroupConvTestCases)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT SSE42 GroupConvolution ============= */
const std::vector<CPUSpecificParams> sse42_GroupConv = {conv_sse42_2D, conv_sse42_2D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_SSE42_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
        //  1. jcp.ur_w (=3,<3)
        //  2. jcp.ur_w_tail (=0,>0)
        //  3. jcp.kw (>7,<=7)
        //  4. jcp.nb_oc = jcp.oc / jcp.oc_block;
        //  5. jcp.nb_ic = jcp.ic / jcp.ic_block;
        //  6. ocb_work

        //  jcp.ur_w == 3, jcp.ur_w_tail == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 10}, 8, 8, sse42_GroupConv, vecPrcConnectParamsFP32),
        //  jcp.ur_w < 3 (jcp.ur_w == jcp.ow)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 4}, 8, 8, sse42_GroupConv, vecPrcConnectParamsFP32),
        //  jcp.ur_w == 3, jcp.ur_w_tail == 0
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 11}, 8, 8, sse42_GroupConv, vecPrcConnectParamsFP32),
        //  jcp.kw > 7
        makeSingleGroupConvCPUTestCases({3, 8}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 10}, 8, 8, sse42_GroupConv, vecPrcConnectParamsFP32),
        //  jcp.nb_oc == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 8, 16, sse42_GroupConv, vecPrcConnectParamsFP32),
        //  jcp.nb_ic == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 16, 8, sse42_GroupConv, vecPrcConnectParamsFP32),
        //  ocb_work > 1 (ocb_work == 2)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 8, 40, sse42_GroupConv, vecPrcConnectParamsFP32),
        //  jcp.nb_ic == 2, ocb_work == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 16, 40, sse42_GroupConv, vecPrcConnectParamsFP32),

        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT,
                                        3, 2, {129, 129}, 8, 8, sse42_GroupConv, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT,
                                        2, 1, {10, 10}, 8, 8, sse42_GroupConv, vecPrcConnectParamsFP32Default)

        //  not supported jit_sse42 for 3d
        //  makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
        //                              3, 2, {33, 33, 33}, 8, 8, cpuParams_sse42_3D),
        //  makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
        //                              2, 1, {10, 10, 10}, 8, 8, cpuParams_sse42_3D),
);

INSTANTIATE_TEST_CASE_P(smoke_JIT_SSE42_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_SSE42_GroupConvTestCases)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX2 GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx2_GroupConv_2D = {conv_avx2_2D, conv_avx2_2D_nspc};
const std::vector<CPUSpecificParams> avx2_GroupConv_3D = {conv_avx2_3D, conv_avx2_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX2_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
        //  1. jcp.ur_w (=3,<3)
        //  2. jcp.ur_w_tail (=0,>0)
        //  3. jcp.kw (>7,<=7)
        //  4. jcp.nb_oc = jcp.oc / jcp.oc_block;
        //  5. jcp.nb_ic = jcp.ic / jcp.ic_block;
        //  6. ocb_work

        //  jcp.ur_w == 3, jcp.ur_w_tail == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 10}, 8, 8, avx2_GroupConv_2D, vecPrcConnectParamsFP32),
        //  jcp.ur_w < 3 (jcp.ur_w == jcp.ow)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 4}, 8, 8, avx2_GroupConv_2D, vecPrcConnectParamsFP32),
        //  jcp.ur_w == 3, jcp.ur_w_tail == 0
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 11}, 8, 8, avx2_GroupConv_2D, vecPrcConnectParamsFP32),
        //  jcp.kw > 7
        makeSingleGroupConvCPUTestCases({3, 8}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 10}, 8, 8, avx2_GroupConv_2D, vecPrcConnectParamsFP32),
        //  jcp.nb_oc == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 8, 16, avx2_GroupConv_2D, vecPrcConnectParamsFP32),
        //  jcp.nb_ic == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 16, 8, avx2_GroupConv_2D, vecPrcConnectParamsFP32),
        //  ocb_work > 1 (ocb_work == 2)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 8, 40, avx2_GroupConv_2D, vecPrcConnectParamsFP32),
        //  jcp.nb_ic == 2, ocb_work == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 16, 40, avx2_GroupConv_2D, vecPrcConnectParamsFP32),

        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT,
                                        3, 2, {129, 129}, 8, 8, avx2_GroupConv_2D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT,
                                        2, 1, {10, 10}, 8, 8, avx2_GroupConv_2D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                        3, 2, {33, 33, 33}, 8, 8, avx2_GroupConv_3D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                        2, 1, {10, 10, 10}, 8, 8, avx2_GroupConv_3D, vecPrcConnectParamsFP32)
);

INSTANTIATE_TEST_CASE_P(smoke_JIT_AVX2_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX2_GroupConvTestCases)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX512 GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx512_GroupConv_2D = {conv_avx512_2D, conv_avx512_2D_nspc};
const std::vector<CPUSpecificParams> avx512_GroupConv_3D = {conv_avx512_3D, conv_avx512_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX512_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
        //  1. "blocked to blocked" or "planar to blocked"
        //  2. jcp.nb_ic, jcp.nb_oc

        //  blocked to blocked
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 16, 16, avx512_GroupConv_2D, vecPrcConnectParams),
        //  jcp.nb_ic == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 32, 16, avx512_GroupConv_2D, vecPrcConnectParams),
        //  jcp.nb_oc == 2
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        2, 1, {5, 5}, 16, 32, avx512_GroupConv_2D, vecPrcConnectParams),

        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 3, 2, {129, 129}, 16, 16,
                                        avx512_GroupConv_2D, vecPrcConnectParams),
        makeSingleGroupConvCPUTestCases({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT,
                                        2, 1, {10, 10}, 16, 16, avx512_GroupConv_2D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                        3, 2, {33, 33, 33}, 16, 16, avx512_GroupConv_3D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                        2, 1, {10, 10, 10}, 16, 16, avx512_GroupConv_3D, vecPrcConnectParams)
);

INSTANTIATE_TEST_CASE_P(smoke_JIT_AVX512_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX512_GroupConvTestCases)),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT SSE42 DW GroupConvolution ============= */
const std::vector<CPUSpecificParams> sse42_DW_2D = {conv_sse42_dw_2D, conv_sse42_dw_2D_nspc};
const std::vector<CPUSpecificParams> sse42_DW_3D = {conv_sse42_dw_3D, conv_sse42_dw_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_SSE42_DW_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
        //  1. jcp.ngroups % simd_w (=0,!=0)
        //  2. jcp.nb_ch
        //  3. jcp.nb_ch_blocking (=2,<2)
        //  4. jcp.ur_w == 3

        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 8)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        8, 1, {5, 5}, 1, 1, sse42_DW_2D, vecPrcConnectParamsFP32),
        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 2, jcp.nb_ch_blocking == 2 (jcp.ngroups == 16)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        16, 1, {5, 5}, 1, 1, sse42_DW_2D, vecPrcConnectParamsFP32),
        //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 3, jcp.nb_ch_blocking == 2 (jcp.ngroups == 17) TODO: pad channels not supported for SSE42
        //  makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
        //  17, 1, {5, 5}, 1, 1, conv_sse42_DW_2D, vecPrcConnectParamsFP32only),
        //  jcp.ow > jcp.ur_w (jcp.ow == 7)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        8, 1, {5, 9}, 1, 1, sse42_DW_2D, vecPrcConnectParamsFP32),

        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 8, 2, {129, 129}, 1, 1,
                                        sse42_DW_2D, vecPrcConnectParamsFP32),
        makeSingleGroupConvCPUTestCases({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT,
                                        8, 1, {10, 10}, 1, 1, sse42_DW_2D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                        8, 2, {33, 33, 33}, 1, 1, sse42_DW_3D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                        8, 1, {10, 10, 10}, 1, 1, sse42_DW_3D, vecPrcConnectParamsFP32)
);

INSTANTIATE_TEST_CASE_P(smoke_JIT_SSE42_DW_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice
(JIT_SSE42_DW_GroupConvTestCases)), GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX2 DW GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx2_DW_2D = {conv_avx2_dw_2D, conv_avx2_dw_2D_nspc};
const std::vector<CPUSpecificParams> avx2_DW_3D = {conv_avx2_dw_3D, conv_avx2_dw_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX2_DW_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
        //  1. jcp.ngroups % simd_w (=0,!=0)
        //  2. jcp.nb_ch
        //  3. jcp.nb_ch_blocking (=3,<3)
        //  4. jcp.ur_w == 4

        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 8)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        8, 1, {5, 5}, 1, 1, avx2_DW_2D, vecPrcConnectParamsFP32),
        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 3, jcp.nb_ch_blocking == 3 (jcp.ngroups == 24)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        24, 1, {5, 5}, 1, 1, avx2_DW_2D, vecPrcConnectParamsFP32),
        //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 4, jcp.nb_ch_blocking == 3 (jcp.ngroups == 25)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        25, 1, {5, 5}, 1, 1, avx2_DW_2D, vecPrcConnectParamsFP32),
        //  jcp.ow > jcp.ur_w (jcp.ow == 7)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        8, 1, {5, 9}, 1, 1, avx2_DW_2D, vecPrcConnectParamsFP32),

        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 8, 2, {129, 129}, 1, 1,
                                        avx2_DW_2D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT,
                                        8, 1, {10, 10}, 1, 1, avx2_DW_2D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                        8, 2, {33, 33, 33}, 1, 1, avx2_DW_3D, vecPrcConnectParamsFP32Default),
        makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                        8, 1, {10, 10, 10}, 1, 1, avx2_DW_3D, vecPrcConnectParamsFP32)
);

INSTANTIATE_TEST_CASE_P(smoke_JIT_AVX2_DW_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice
(JIT_AVX2_DW_GroupConvTestCases)), GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT AVX512 DW GroupConvolution ============= */
const std::vector<CPUSpecificParams> avx512_DW_2D = {conv_avx512_dw_2D, conv_avx512_dw_2D_nspc};
const std::vector<CPUSpecificParams> avx512_DW_3D = {conv_avx512_dw_3D, conv_avx512_dw_3D_nspc};
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX512_DW_GroupConvTestCases = generateSingleGroupConvCPUTestCases(
        //  1. jcp.ngroups % simd_w (=0,!=0)
        //  2. jcp.nb_ch
        //  3. jcp.nb_ch_blocking (=4,<4)
        //  4. jcp.ur_w == 6

        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 16)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        16, 1, {5, 5}, 1, 1, avx512_DW_2D, vecPrcConnectParams),
        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 4, jcp.nb_ch_blocking == 4 (jcp.ngroups == 64)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        64, 1, {5, 5}, 1, 1, avx512_DW_2D, vecPrcConnectParams),
        //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 5, jcp.nb_ch_blocking == 4 (jcp.ngroups == 65)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        65, 1, {5, 5}, 1, 1, avx512_DW_2D, vecPrcConnectParams),
        //  jcp.ow > jcp.ur_w (jcp.ow == 7)
        makeSingleGroupConvCPUTestCases({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID,
                                        8, 1, {5, 9}, 1, 1, avx512_DW_2D, vecPrcConnectParams),
        //  "hard" cases
        makeSingleGroupConvCPUTestCases({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 16, 2, {129, 129}, 1, 1,
                                        avx512_DW_2D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 16, 1, {10, 10}, 1, 1,
                                        avx512_DW_2D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                        16, 2, {33, 33, 33}, 1, 1, avx512_DW_3D, vecPrcConnectParamsDefault),
        makeSingleGroupConvCPUTestCases({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                        16, 1, {10, 10, 10}, 1, 1, avx512_DW_3D, vecPrcConnectParams)
);

INSTANTIATE_TEST_CASE_P(smoke_JIT_AVX512_DW_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice
(JIT_AVX512_DW_GroupConvTestCases)), GroupConvolutionLayerCPUTest::getTestCaseName);

/* ============= JIT SSE42 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX2 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX512 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX2 PLANAR Convolution (not supported with groups) ============= */
/* ============= JIT AVX5122 PLANAR Convolution (not supported with groups) ============= */
/* ============================================= */

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
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::ValuesIn(numGroups_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1D = {
        conv_sse42_1D,
        conv_avx2_1D,
        conv_avx512_1D
};

INSTANTIATE_TEST_CASE_P(smoke_GroupConv_1D, GroupConvolutionLayerCPUTest,
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
        ::testing::Values(fusingAddPerChannel)),
    GroupConvolutionLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
