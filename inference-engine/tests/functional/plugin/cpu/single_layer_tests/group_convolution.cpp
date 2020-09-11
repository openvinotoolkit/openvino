// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/group_convolution.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        groupConvLayerTestParamsSet,
        CPUSpecificParams> groupConvLayerCPUTestParamsSet;

class GroupConvolutionLayerCPUTest : public testing::WithParamInterface<groupConvLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvLayerCPUTestParamsSet> obj) {
        groupConvLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::GroupConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<groupConvLayerTestParamsSet>(
                basicParamsSet, 0));

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        groupConvLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        groupConvSpecificParams groupConvParams;
        std::vector<size_t> inputShape;
        auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(groupConvParams, netPrecision, inputShape, targetDevice) = basicParamsSet;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        auto groupConv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(
                ngraph::builder::makeGroupConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                      padEnd, dilation, padType, convOutChannels, numGroups));
        groupConv->get_rt_info() = setCPUInfo(inFmts, outFmts, priority);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(groupConv)};
        function = std::make_shared<ngraph::Function>(results, params, "groupConvolution");
    }
};

TEST_P(GroupConvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckCPUImpl(executableNetwork, "Convolution", inFmts, outFmts, selectedType);
}

namespace {

/* GROUP CONV TEST UTILS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(std::vector<CPUSpecificParams> CPUParams) {
    std::vector<CPUSpecificParams> resCPUParams;
    const int selectedTypeIndex = 3;

    for (auto param : CPUParams) {
        auto selectedTypeStr = std::get<selectedTypeIndex>(param);

        if (selectedTypeStr.find("jit") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("sse42") != std::string::npos && !with_cpu_x86_sse42())
            continue;
        if (selectedTypeStr.find("avx2") != std::string::npos && !with_cpu_x86_avx2())
            continue;
        if (selectedTypeStr.find("avx512") != std::string::npos && !with_cpu_x86_avx512f())
            continue;

        resCPUParams.push_back(param);
    }

    return resCPUParams;
}

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

const std::vector<CPUSpecificParams> CPUParams_Planar_2D = {
        conv_gemm_2D
};

INSTANTIATE_TEST_CASE_P(GroupConv_2D_Planar_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Planar_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >({2, 12, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_2D))),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

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

const std::vector<CPUSpecificParams> CPUParams_Planar_3D = {
        conv_gemm_3D
};

INSTANTIATE_TEST_CASE_P(GroupConv_3D_Planar_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Planar_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >({2, 12, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_3D))),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

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

const std::vector<CPUSpecificParams> CPUParams_Blocked_2D = {
        conv_sse42_2D,
        conv_avx2_2D,
        conv_avx512_2D
};

INSTANTIATE_TEST_CASE_P(GroupConv_2D_Blocked_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Blocked_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >({2, 64, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_2D))),
                        GroupConvolutionLayerCPUTest::getTestCaseName);

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

const std::vector<CPUSpecificParams> CPUParams_Blocked_3D = {
//        conv_sse42_3D, // not supported jit_sse42 for 3d
        conv_avx2_3D,
        conv_avx512_3D
};

INSTANTIATE_TEST_CASE_P(GroupConv_3D_Blocked_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Blocked_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >({2, 64, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_3D))),
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
        conv_avx512_dw_2D
};

INSTANTIATE_TEST_CASE_P(GroupConv_2D_DW_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >({2, 32, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_2D))),
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
        conv_avx512_dw_3D
};

INSTANTIATE_TEST_CASE_P(GroupConv_3D_DW_FP32, GroupConvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >({2, 32, 7, 7, 7})),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_3D))),
                        GroupConvolutionLayerCPUTest::getTestCaseName);
/* ========= */


/* ============= SINGLE TEST CASES ============= */
groupConvLayerCPUTestParamsSet makeSingleGroupConvCPUTestCase(SizeVector kernels, SizeVector strides, SizeVector dilations,
                                                        std::vector<ptrdiff_t> padBegins, std::vector<ptrdiff_t> padEnds, ngraph::op::PadType padType,
                                                        int groups, int mb, SizeVector spDims, int inGroupSize, int outGroupSize,
                                                        CPUSpecificParams CPUParams) {
    int inChannels = groups * inGroupSize;
    int outChannels = groups * outGroupSize;

    SizeVector inputShapes;
    inputShapes.push_back(mb);
    inputShapes.push_back(inChannels);
    inputShapes.insert(inputShapes.end(), spDims.begin(), spDims.end());

    groupConvSpecificParams specificParams(kernels, strides, padBegins, padEnds, dilations, outChannels, groups, padType);
    groupConvLayerTestParamsSet basicParamsSet(specificParams, Precision::FP32, inputShapes, CommonTestUtils::DEVICE_CPU);
    return groupConvLayerCPUTestParamsSet(basicParamsSet, CPUParams);
}

/* ============= GEMM GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> gemmGroupConvTestCases = {
        //  1. is_depthwise (true, false)
        //  2. jcp.im2col_sz (=0,>0)
        //  3. is_blocking_applicable (true, false)

        //  is_depthwise == false, im2col_sz > 0
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 2, 2, conv_gemm_2D),
        //  is_depthwise == true
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 1, 1, conv_gemm_2D),
        //  im2col_sz == 0, is_blocking_applicable == true
        makeSingleGroupConvCPUTestCase({1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 2, 2, conv_gemm_2D),
        //  is_blocking_applicable == false ((jcp.im2col_sz == 0) && (jcp.ic / jcp.oc >= 42))
        makeSingleGroupConvCPUTestCase({1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 42, 1, conv_gemm_2D),

        //  "hard" cases
        makeSingleGroupConvCPUTestCase({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 3, 2, {129, 129}, 4, 2, conv_gemm_2D),
        makeSingleGroupConvCPUTestCase({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 2, 1, {10, 10}, 3, 3, conv_gemm_2D),
        makeSingleGroupConvCPUTestCase({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                3, 2, {33, 33, 33}, 4, 2, conv_gemm_3D),
        makeSingleGroupConvCPUTestCase({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                2, 1, {10, 10, 10}, 3, 3, conv_gemm_3D),
};

INSTANTIATE_TEST_CASE_P(GEMM_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(gemmGroupConvTestCases)));

/* ============= JIT SSE42 GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> JIT_SSE42_GroupConvTestCases = {
        //  1. jcp.ur_w (=3,<3)
        //  2. jcp.ur_w_tail (=0,>0)
        //  3. jcp.kw (>7,<=7)
        //  4. jcp.nb_oc = jcp.oc / jcp.oc_block;
        //  5. jcp.nb_ic = jcp.ic / jcp.ic_block;
        //  6. ocb_work

        //  jcp.ur_w == 3, jcp.ur_w_tail == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 10}, 8, 8, conv_sse42_2D),
        //  jcp.ur_w < 3 (jcp.ur_w == jcp.ow)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 4}, 8, 8, conv_sse42_2D),
        //  jcp.ur_w == 3, jcp.ur_w_tail == 0
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 11}, 8, 8, conv_sse42_2D),
        //  jcp.kw > 7
        makeSingleGroupConvCPUTestCase({3, 8}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 10}, 8, 8, conv_sse42_2D),
        //  jcp.nb_oc == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 8, 16, conv_sse42_2D),
        //  jcp.nb_ic == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 16, 8, conv_sse42_2D),
        //  ocb_work > 1 (ocb_work == 2)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 8, 40, conv_sse42_2D),
        //  jcp.nb_ic == 2, ocb_work == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 16, 40, conv_sse42_2D),

        //  "hard" cases
        makeSingleGroupConvCPUTestCase({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 3, 2, {129, 129}, 8, 8, conv_sse42_2D),
        makeSingleGroupConvCPUTestCase({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 2, 1, {10, 10}, 8, 8, conv_sse42_2D),

        //  not supported jit_sse42 for 3d
        //  makeSingleGroupConvCPUTestCase({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
        //                              3, 2, {33, 33, 33}, 8, 8, cpuParams_sse42_3D),
        //  makeSingleGroupConvCPUTestCase({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
        //                              2, 1, {10, 10, 10}, 8, 8, cpuParams_sse42_3D),
};

INSTANTIATE_TEST_CASE_P(JIT_SSE42_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_SSE42_GroupConvTestCases)));

/* ============= JIT AVX2 GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX2_GroupConvTestCases = {
        //  1. jcp.ur_w (=3,<3)
        //  2. jcp.ur_w_tail (=0,>0)
        //  3. jcp.kw (>7,<=7)
        //  4. jcp.nb_oc = jcp.oc / jcp.oc_block;
        //  5. jcp.nb_ic = jcp.ic / jcp.ic_block;
        //  6. ocb_work

        //  jcp.ur_w == 3, jcp.ur_w_tail == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 10}, 8, 8, conv_avx2_2D),
        //  jcp.ur_w < 3 (jcp.ur_w == jcp.ow)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 4}, 8, 8, conv_avx2_2D),
        //  jcp.ur_w == 3, jcp.ur_w_tail == 0
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 11}, 8, 8, conv_avx2_2D),
        //  jcp.kw > 7
        makeSingleGroupConvCPUTestCase({3, 8}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 10}, 8, 8, conv_avx2_2D),
        //  jcp.nb_oc == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 8, 16, conv_avx2_2D),
        //  jcp.nb_ic == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 16, 8, conv_avx2_2D),
        //  ocb_work > 1 (ocb_work == 2)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 8, 40, conv_avx2_2D),
        //  jcp.nb_ic == 2, ocb_work == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 16, 40, conv_avx2_2D),

        //  "hard" cases
        makeSingleGroupConvCPUTestCase({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 3, 2, {129, 129}, 8, 8, conv_avx2_2D),
        makeSingleGroupConvCPUTestCase({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 2, 1, {10, 10}, 8, 8, conv_avx2_2D),
        makeSingleGroupConvCPUTestCase({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                    3, 2, {33, 33, 33}, 8, 8, conv_avx2_3D),
        makeSingleGroupConvCPUTestCase({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                    2, 1, {10, 10, 10}, 8, 8, conv_avx2_3D),
};

INSTANTIATE_TEST_CASE_P(JIT_AVX2_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX2_GroupConvTestCases)));

/* ============= JIT AVX512 GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX512_GroupConvTestCases = {
        //  1. "blocked to blocked" or "planar to blocked"
        //  2. jcp.nb_ic, jcp.nb_oc

        //  blocked to blocked
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 16, 16, conv_avx512_2D),
        //  jcp.nb_ic == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 32, 16, conv_avx512_2D),
        //  jcp.nb_oc == 2
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 2, 1, {5, 5}, 16, 32, conv_avx512_2D),

        //  "hard" cases
        makeSingleGroupConvCPUTestCase({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 3, 2, {129, 129}, 16, 16,
                                       conv_avx512_2D),
        makeSingleGroupConvCPUTestCase({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 2, 1, {10, 10}, 16, 16, conv_avx512_2D),
        makeSingleGroupConvCPUTestCase({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                    3, 2, {33, 33, 33}, 16, 16, conv_avx512_3D),
        makeSingleGroupConvCPUTestCase({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                    2, 1, {10, 10, 10}, 16, 16, conv_avx512_3D),
};

INSTANTIATE_TEST_CASE_P(JIT_AVX512_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX512_GroupConvTestCases)));

/* ============= JIT SSE42 DW GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> JIT_SSE42_DW_GroupConvTestCases = {
        //  1. jcp.ngroups % simd_w (=0,!=0)
        //  2. jcp.nb_ch
        //  3. jcp.nb_ch_blocking (=2,<2)
        //  4. jcp.ur_w == 3

        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 8)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 8, 1, {5, 5}, 1, 1, conv_sse42_dw_2D),
        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 2, jcp.nb_ch_blocking == 2 (jcp.ngroups == 16)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 16, 1, {5, 5}, 1, 1, conv_sse42_dw_2D),
        //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 3, jcp.nb_ch_blocking == 2 (jcp.ngroups == 17) TODO: pad channels not supported for SSE42
        //  makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 17, 1, {5, 5}, 1, 1, conv_sse42_dw_2D),
        //  jcp.ow > jcp.ur_w (jcp.ow == 7)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 8, 1, {5, 9}, 1, 1, conv_sse42_dw_2D),

        //  "hard" cases
        makeSingleGroupConvCPUTestCase({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 8, 2, {129, 129}, 1, 1,
                                       conv_sse42_dw_2D),
        makeSingleGroupConvCPUTestCase({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 8, 1, {10, 10}, 1, 1, conv_sse42_dw_2D),
        makeSingleGroupConvCPUTestCase({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                    8, 2, {33, 33, 33}, 1, 1, conv_sse42_dw_3D),
        makeSingleGroupConvCPUTestCase({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                    8, 1, {10, 10, 10}, 1, 1, conv_sse42_dw_3D),
};

INSTANTIATE_TEST_CASE_P(JIT_SSE42_DW_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_SSE42_DW_GroupConvTestCases)));

/* ============= JIT AVX2 DW GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX2_DW_GroupConvTestCases = {
        //  1. jcp.ngroups % simd_w (=0,!=0)
        //  2. jcp.nb_ch
        //  3. jcp.nb_ch_blocking (=3,<3)
        //  4. jcp.ur_w == 4

        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 8)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 8, 1, {5, 5}, 1, 1, conv_avx2_dw_2D),
        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 3, jcp.nb_ch_blocking == 3 (jcp.ngroups == 24)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 24, 1, {5, 5}, 1, 1, conv_avx2_dw_2D),
        //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 4, jcp.nb_ch_blocking == 3 (jcp.ngroups == 25)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 25, 1, {5, 5}, 1, 1, conv_avx2_dw_2D),
        //  jcp.ow > jcp.ur_w (jcp.ow == 7)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 8, 1, {5, 9}, 1, 1, conv_avx2_dw_2D),

        //  "hard" cases
        makeSingleGroupConvCPUTestCase({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 8, 2, {129, 129}, 1, 1,
                                       conv_avx2_dw_2D),
        makeSingleGroupConvCPUTestCase({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 8, 1, {10, 10}, 1, 1, conv_avx2_dw_2D),
        makeSingleGroupConvCPUTestCase({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                    8, 2, {33, 33, 33}, 1, 1, conv_avx2_dw_3D),
        makeSingleGroupConvCPUTestCase({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                    8, 1, {10, 10, 10}, 1, 1, conv_avx2_dw_3D),
};

INSTANTIATE_TEST_CASE_P(JIT_AVX2_DW_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX2_DW_GroupConvTestCases)));

/* ============= JIT AVX512 DW GroupConvolution ============= */
const std::vector<groupConvLayerCPUTestParamsSet> JIT_AVX512_DW_GroupConvTestCases = {
        //  1. jcp.ngroups % simd_w (=0,!=0)
        //  2. jcp.nb_ch
        //  3. jcp.nb_ch_blocking (=4,<4)
        //  4. jcp.ur_w == 6

        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 1, jcp.nb_ch_blocking == 1 (jcp.ngroups == 16)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 16, 1, {5, 5}, 1, 1, conv_avx512_dw_2D),
        //  jcp.ngroups % simd_w == 0, jcp.nb_ch == 4, jcp.nb_ch_blocking == 4 (jcp.ngroups == 64)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 64, 1, {5, 5}, 1, 1, conv_avx512_dw_2D),
        //  jcp.ngroups % simd_w != 0, jcp.nb_ch == 5, jcp.nb_ch_blocking == 4 (jcp.ngroups == 65)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 65, 1, {5, 5}, 1, 1, conv_avx512_dw_2D),
        //  jcp.ow > jcp.ur_w (jcp.ow == 7)
        makeSingleGroupConvCPUTestCase({3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, ngraph::op::PadType::VALID, 8, 1, {5, 9}, 1, 1, conv_avx512_dw_2D),

        //  "hard" cases
        makeSingleGroupConvCPUTestCase({3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 16, 2, {129, 129}, 1, 1,
                                       conv_avx512_dw_2D),
        makeSingleGroupConvCPUTestCase({2, 4}, {1, 2}, {3, 2}, {2, 1}, {1, 0}, ngraph::op::PadType::EXPLICIT, 16, 1, {10, 10}, 1, 1,
                                       conv_avx512_dw_2D),
        makeSingleGroupConvCPUTestCase({3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, ngraph::op::PadType::EXPLICIT,
                                    16, 2, {33, 33, 33}, 1, 1, conv_avx512_dw_3D),
        makeSingleGroupConvCPUTestCase({2, 3, 4}, {1, 2, 2}, {3, 1, 2}, {2, 2, 1}, {1, 1, 0}, ngraph::op::PadType::EXPLICIT,
                                    16, 1, {10, 10, 10}, 1, 1, conv_avx512_dw_3D),
};

INSTANTIATE_TEST_CASE_P(JIT_AVX512_DW_GroupConv, GroupConvolutionLayerCPUTest, ::testing::ValuesIn(filterParamsSetForDevice(JIT_AVX512_DW_GroupConvTestCases)));

/* ============= JIT SSE42 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX2 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX512 1x1 Convolution (not supported with groups) ============= */
/* ============= JIT AVX2 PLANAR Convolution (not supported with groups) ============= */
/* ============= JIT AVX5122 PLANAR Convolution (not supported with groups) ============= */
/* ============================================= */

} // namespace

} // namespace CPULayerTestsDefinitions
