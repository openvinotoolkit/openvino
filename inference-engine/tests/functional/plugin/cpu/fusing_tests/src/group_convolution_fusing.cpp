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
    std::vector<std::shared_ptr<ngraph::Node>> postNodes;
    std::vector<std::string> fusedOps;
    std::tie(postFunction, postNodes, fusedOps) = fusingParams;

    if (postFunction) {
        result << "_Fused=" << postFunction->get_friendly_name();
    } else {
        result << "_Fused=" << FusingTestUtils::postNodes2str(postNodes);
    }

    return result.str();
}

void GroupConvolutionLayerFusingTest::SetUp() {
    groupConvLayerTestParamsSet basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postFunction, postNodes, fusedOps) = fusingParams;

    groupConvSpecificParams groupConvParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(groupConvParams, netPrecision, inputShape, targetDevice) = basicParamsSet;

    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ngraph::ParameterVector params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto groupConv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(
            ngraph::builder::makeGroupConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                  padEnd, dilation, padType, convOutChannels, numGroups));
    groupConv->get_rt_info() = CPUTestUtils::setCPUInfo(inFmts, outFmts, priority);

    if (postFunction) {
        function = makeNgraphFunction(ngPrc, params, groupConv, postFunction);
    } else {
        function = makeNgraphFunction(ngPrc, params, groupConv, postNodes);
    }
}

TEST_P(GroupConvolutionLayerFusingTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CPUTestUtils::CheckCPUImpl(executableNetwork, "Convolution", inFmts, outFmts, selectedType);
    FusingTestUtils::CheckFusing(executableNetwork, "Convolution", fusedOps);
}

namespace {

std::vector<fusingSpecificParams> fusingParamsSet {
    // activations
    fusingRelu,
    fusingElu,
    fusingSigmoid,
    fusingClamp,
    fusingPRelu,
    // other patterns
    fusingReluScaleShift,
//    fusingFakeQuantizeRelu, // todo: failed test
    fusingSum,
};

std::vector<fusingSpecificParams> makeFusingParamsSetWithShape(std::vector<size_t> shape) {
    std::vector<fusingSpecificParams> paramsSet;

    paramsSet.push_back({makeSwishPattern(shape), {}, {"Swish"}});
    paramsSet.push_back({makeFakeQuantizeActivationPattern(256, ngraph::helpers::Relu, shape), {}, {"FakeQuantize", "Relu"}});

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
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_HardCases_Planar_2D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Planar_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 10, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_2D)),
                                ::testing::ValuesIn(makeFusingParamsSetWithShape({1, 4, 3, 3}))),
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
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_HardCases_Planar_3D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Planar_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 10, 5, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Planar_3D)),
                                ::testing::ValuesIn(makeFusingParamsSetWithShape({1, 4, 3, 3, 3}))),
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
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_HardCases_Blocked_2D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Blocked_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_2D)),
                                ::testing::ValuesIn(makeFusingParamsSetWithShape({1, 4, 3, 3}))),
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
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_HardCases_Blocked_3D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_Blocked_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Blocked_3D)),
                                ::testing::ValuesIn(makeFusingParamsSetWithShape({1, 4, 3, 3, 3}))),
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
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_HardCases_DW_2D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_2D)),
                                ::testing::ValuesIn(makeFusingParamsSetWithShape({1, 32, 3, 3}))),
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

INSTANTIATE_TEST_CASE_P(GroupConvFusing_DW_3D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_3D)),
                                ::testing::ValuesIn(fusingParamsSet)),
                        GroupConvolutionLayerFusingTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvFusing_HardCases_DW_3D, GroupConvolutionLayerFusingTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        groupConvParams_ExplicitPadding_DW_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(std::vector<size_t >{1, 32, 5, 5, 5}),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_DW_3D)),
                                ::testing::ValuesIn(makeFusingParamsSetWithShape({1, 32, 3, 3, 3}))),
                        GroupConvolutionLayerFusingTest::getTestCaseName);
/* ========= */

} // namespace

} // namespace CPUFusingTestsDefinitions