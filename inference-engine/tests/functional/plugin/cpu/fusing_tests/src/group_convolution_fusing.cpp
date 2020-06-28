// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/group_convolution_fusing.hpp"

using namespace CPUTestUtils::GroupConv;

namespace CPUFusingTestsDefinitions {

std::string GroupConvolutionLayerFusingTest::getTestCaseName(testing::TestParamInfo<groupConvLayerFusingTestParamsSet> obj) {
    groupConvLayerTestParamsSet basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

    std::ostringstream result;
    result << LayerTestsDefinitions::GroupConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<groupConvLayerTestParamsSet>(
            basicParamsSet, 0));

    std::vector<cpu_memory_format_t> inFmts, outFmts;
    std::vector<std::string> priority;
    std::string selectedType;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    result << "_inFmts=" << CPUTestUtils::fmts2str(inFmts);
    result << "_outFmts=" << CPUTestUtils::fmts2str(outFmts);
    result << "_primitive=" << selectedType;

    std::shared_ptr<ngraph::Function> postFunction;
    std::vector<std::string> fusedOps;
    std::tie(postFunction, fusedOps) = fusingParams;

    result << "_Fused=" << postFunction->get_friendly_name();

    return result.str();
}

void GroupConvolutionLayerFusingTest::SetUp() {
    groupConvLayerTestParamsSet basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postFunction, fusedOps) = fusingParams;

    groupConvSpecificParams groupConvParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(groupConvParams, netPrecision, inputShape, targetDevice) = basicParamsSet;

    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

    auto clonedPostFunction = clone_function(*postFunction);
    clonedPostFunction->set_friendly_name(postFunction->get_friendly_name());

    auto patternName = clonedPostFunction->get_friendly_name();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ngraph::ParameterVector params;
    if (patternName == "Sum") {
        params = ngraph::builder::makeParams(ngPrc, {inputShape, clonedPostFunction->get_parameters()[1]->get_partial_shape().get_shape()});
    } else {
        params = ngraph::builder::makeParams(ngPrc, {inputShape});
    }

    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto groupConv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(
            ngraph::builder::makeGroupConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                  padEnd, dilation, padType, convOutChannels, numGroups));
    groupConv->get_rt_info() = CPUTestUtils::setCPUInfo(inFmts, outFmts, priority);

    clonedPostFunction->replace_node(clonedPostFunction->get_parameters()[0], groupConv);
    if (patternName == "Sum") {
        clonedPostFunction->replace_node(clonedPostFunction->get_parameters()[1], params[1]);
    }
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(clonedPostFunction->get_result()->get_input_node_shared_ptr(0))};

    function = std::make_shared<ngraph::Function>(results, params, "groupConvolutionFusing");
}

TEST_P(GroupConvolutionLayerFusingTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CPUTestUtils::CheckCPUImpl(executableNetwork, "Convolution", inFmts, outFmts, selectedType);
    FusingTestUtils::CheckFusing(executableNetwork, "Convolution", fusedOps);
}

namespace {

std::vector<fusingSpecificParams> fusingParamsSet(std::vector<size_t> shape) {
    std::vector<fusingSpecificParams> paramsSet;

    // activations
    paramsSet.push_back({makeActivationPattern(shape, ngraph::helpers::Relu), {"Relu"}});
    paramsSet.push_back({makeActivationPattern(shape, ngraph::helpers::Elu, 2.0f), {"Elu"}});
    paramsSet.push_back({makeActivationPattern(shape, ngraph::helpers::Sigmoid), {"Sigmoid"}});
    paramsSet.push_back({makeActivationPattern(shape, ngraph::helpers::Clamp, 3.0f, 6.0f), {"Clamp"}});
    paramsSet.push_back({makeActivationPattern(shape, ngraph::helpers::LeakyRelu), {"PRelu"}});

    // other patterns
    paramsSet.push_back({makeSwishPattern(shape), {"Swish"}});
    paramsSet.push_back({makeActivationScaleShiftPattern(ngraph::helpers::Relu, shape), {"Add"}});
    paramsSet.push_back({makeFakeQuantizeActivationPattern(256, ngraph::helpers::Relu, shape), {"FakeQuantize", "Relu"}});
    paramsSet.push_back({makeSumPattern(shape), {"Add"}});

    return paramsSet;
    }

/* INSTANCES */
/* ============= GroupConvolution (Planar 2D) ============= */
const auto groupConvParams_ExplicitPadding_Planar_2D = ::testing::Combine(
        ::testing::Values(SizeVector({3, 3})),           //  kernels
        ::testing::Values(SizeVector{1, 1}),             //  strides
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}), //  padBegins
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}), //  padEnds
        ::testing::Values(SizeVector{1, 1}),             //  dilations
        ::testing::Values(4),                            //  numOutChannels
        ::testing::Values(2),                            //  numGroups
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_Planar_2D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Planar_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 10, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_2D)),
                                ::testing::ValuesIn(fusingParamsSet({1, 4, 3, 3}))),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

/* ============= GroupConvolution (Planar 3D) ============= */
const auto groupConvParams_ExplicitPadding_Planar_3D = ::testing::Combine(
        ::testing::Values(SizeVector({3, 3, 3})),           //  kernels
        ::testing::Values(SizeVector{1, 1, 1}),             //  strides
        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}), //  padBegins
        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}), //  padEnds
        ::testing::Values(SizeVector{1, 1, 1}),             //  dilations
        ::testing::Values(4),                               //  numOutChannels
        ::testing::Values(2),                               //  numGroups
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_Planar_3D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Planar_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 10, 5, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_3D)),
                                ::testing::ValuesIn(fusingParamsSet({1, 4, 3, 3, 3}))),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 2D) ============= */
const auto groupConvParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
        ::testing::Values(SizeVector({3, 3})),           //  kernels
        ::testing::Values(SizeVector{1, 1}),             //  strides
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}), //  padBegins
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}), //  padEnds
        ::testing::Values(SizeVector{1, 1}),             //  dilations
        ::testing::Values(32),                           //  numOutChannels
        ::testing::Values(2),                            //  numGroups
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_Blocked_2D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Blocked_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_2D)),
                                ::testing::ValuesIn(fusingParamsSet({1, 32, 3, 3}))),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

/* ============= GroupConvolution (Blocked 3D) ============= */
const auto groupConvParams_ExplicitPadding_Blocked_3D = ::testing::Combine(
        ::testing::Values(SizeVector({3, 3, 3})),           //  kernels
        ::testing::Values(SizeVector{1, 1, 1}),             //  strides
        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}), //  padBegins
        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}), //  padEnds
        ::testing::Values(SizeVector{1, 1, 1}),             //  dilations
        ::testing::Values(32),                              //  numOutChannels
        ::testing::Values(2),                               //  numGroups
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_Blocked_3D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Blocked_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_3D)),
                                ::testing::ValuesIn(fusingParamsSet({1, 32, 3, 3, 3}))),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

/* ============= GroupConvolution (DW 2D) ============= */
const auto groupConvParams_ExplicitPadding_DW_2D = ::testing::Combine(
        ::testing::Values(SizeVector({3, 3})),           //  kernels
        ::testing::Values(SizeVector{1, 1}),             //  strides
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}), //  padBegins
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}), //  padEnds
        ::testing::Values(SizeVector{1, 1}),             //  dilations
        ::testing::Values(32),                           //  numOutChannels
        ::testing::Values(32),                           //  numGroups
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_DW_2D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_2D)),
                                ::testing::ValuesIn(fusingParamsSet({1, 32, 3, 3}))),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

/* ============= GroupConvolution (DW 3D) ============= */
const auto groupConvParams_ExplicitPadding_DW_3D = ::testing::Combine(
        ::testing::Values(SizeVector({3, 3, 3})),           //  kernels
        ::testing::Values(SizeVector{1, 1, 1}),             //  strides
        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}), //  padBegins
        ::testing::Values(std::vector<ptrdiff_t>{0, 0, 0}), //  padEnds
        ::testing::Values(SizeVector{1, 1, 1}),             //  dilations
        ::testing::Values(32),                              //  numOutChannels
        ::testing::Values(32),                              //  numGroups
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_DW_3D = {
        cpuParams_sse42_dw_3D,
        cpuParams_avx2_dw_3D,
        cpuParams_avx512_dw_3D
};

INSTANTIATE_TEST_CASE_P(GroupConvFusing_DW_3D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_3D)),
                                ::testing::ValuesIn(fusingParamsSet({1, 32, 3, 3, 3}))),
                        GroupConvolutionLayerFusingTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPUFusingTestsDefinitions