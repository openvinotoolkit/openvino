// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <shared_test_classes/single_layer/convolution_backprop.hpp>


using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
using LayerTestsDefinitions::convBackpropSpecificParams;
using LayerTestsDefinitions::convBackpropLayerTestParamsSet;

typedef std::tuple<
    convBackpropLayerTestParamsSet,
    CPUSpecificParams,
    fusingSpecificParams,
    std::map<std::string, std::string> > deconvLayerCPUTestParamsSet;

class DeconvolutionLayerCPUTest : public testing::WithParamInterface<deconvLayerCPUTestParamsSet>,
    virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<deconvLayerCPUTestParamsSet> obj) {
        convBackpropLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ConvolutionBackpropLayerTest::getTestCaseName(testing::TestParamInfo<convBackpropLayerTestParamsSet>(
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
        using namespace ngraph;
        convBackpropLayerTestParamsSet basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        convBackpropSpecificParams convParams;
        std::vector<size_t> inputShape;
        std::vector<size_t> outputShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outputShape, targetDevice) = basicParamsSet;

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
        std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = convParams;
        auto inElementType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto outElementType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);
        if (inPrc == Precision::BF16)
            inElementType = element::f32;
        if (outPrc == Precision::BF16)
            outElementType = element::f32;

        auto inputParams = builder::makeParams(inElementType, { inputShape });
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        auto weiPrc = (inElementType == element::u8) ? element::i8 : inElementType;

        auto deconvolutionNode = builder::makeConvolutionBackpropDataRelaxed(paramOuts.front(), weiPrc, outElementType,
                kernel, stride, padBegin, padEnd, dilation, padType, convOutChannels);

        // todo:
        if (!outputShape.empty()) {
            auto outShape = ngraph::opset3::Constant::create(ngraph::element::i64, {outputShape.size()}, outputShape);
            deconvolutionNode = ngraph::builder::makeConvolutionBackpropData(paramOuts.front(), outShape, weiPrc, kernel, stride, padBegin,
                padEnd, dilation, padType, convOutChannels);
        }

        function = makeNgraphFunction(element::f32, inputParams, deconvolutionNode, "convolutionBackpropData");

        if (inPrc == Precision::U8 || inPrc == Precision::I8) {
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::i8, element::f32>>());
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::u8, element::f32>>());
        }
        if (outPrc != Precision::FP32 && outPrc != Precision::BF16) {
            additionalPasses.push_back(std::make_shared<ConvertPrecision<opset1::ConvolutionBackpropData>>());
        }
    }
};

TEST_P(DeconvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Deconvolution");
}

namespace {

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        fusingScaleShift
};

const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES } };
const std::vector<SizeVector> emptyOutputShape = { {} };
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = { {} };

/* ============= Deconvolution params (planar layout) ============= */
const SizeVector numOutChannels_Planar = { 6 };

/* ============= Deconvolution params (blocked layout) ============= */
const SizeVector numOutChannels_Blocked = { 64 };

/* ============= Deconvolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1} };

/* ============= Deconvolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<SizeVector> dilations3d = { {1, 1, 1} };
/* ============= */

/* INSTANCES */
/* ============= Deconvolution (Planar 2D) ============= */
const auto convParams_ExplicitPadding_Planar_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution (Planar 3D) ============= */
const auto convParams_ExplicitPadding_Planar_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution (Blocked 2D) ============= */
const auto convParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution (Blocked 3D) ============= */
const auto convParams_ExplicitPadding_Blocked_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */

const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_1x1_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_1x1_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_1x1_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_1x1_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::ValuesIn(emptyOutputShape),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution params I8 (2D) ============= */
const std::vector<SizeVector> kernels2di8 = { {3, 3} };
const std::vector<SizeVector> strides2di8 = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2di8 = { {0, 0}, {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds2di8 = { {0, 0}, {1, 1} };
const std::vector<SizeVector> dilations2di8 = { {1, 1}/*, {2, 2}*/ };

const auto deconvParams_2D_I8 = ::testing::Combine(
        ::testing::ValuesIn(kernels2di8),
        ::testing::ValuesIn(strides2di8),
        ::testing::ValuesIn(padBegins2di8),
        ::testing::ValuesIn(padEnds2di8),
        ::testing::ValuesIn(dilations2di8),
        ::testing::ValuesIn(numOutChannels_Planar),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

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

const std::vector<CPUSpecificParams> CPUParams_2D_I8 = {
        conv_sse42_2D_nspc,
        conv_avx2_2D_nspc,
        conv_avx512_2D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_I8, DeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        deconvParams_2D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
                                        ::testing::ValuesIn(emptyOutputShape),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution params I8 (3D) ============= */
const std::vector<SizeVector> kernels3di8 = { {3, 3, 3} };
const std::vector<SizeVector> strides3di8 = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3di8 = { {0, 0, 0}, {1, 1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds3di8 = { {0, 0, 0}, {1, 1, 1} };
const std::vector<SizeVector> dilations3di8 = { {1, 1, 1}/*, {2, 2, 2}*/ };

const auto deconvParams_3D_I8 = ::testing::Combine(
        ::testing::ValuesIn(kernels3di8),
        ::testing::ValuesIn(strides3di8),
        ::testing::ValuesIn(padBegins3di8),
        ::testing::ValuesIn(padEnds3di8),
        ::testing::ValuesIn(dilations3di8),
        ::testing::ValuesIn(numOutChannels_Planar),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<CPUSpecificParams> CPUParams_3D_I8 = {
        conv_sse42_3D_nspc,
        conv_avx2_3D_nspc,
        conv_avx512_3D_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_I8, DeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        deconvParams_3D_I8,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
                                        ::testing::ValuesIn(emptyOutputShape),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        DeconvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_2D_1x1_I8 = {
        conv_sse42_2D_1x1_nspc,
        conv_avx2_2D_1x1_nspc,
        conv_avx512_2D_1x1_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_1x1_I8, DeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        convParams_ExplicitPadding_1x1_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
                                        ::testing::ValuesIn(emptyOutputShape),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_1x1_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (3D) ============= */

const auto convParams_ExplicitPadding_1x1_3D = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1, 1})),
        ::testing::Values(SizeVector({1, 1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(SizeVector({1, 1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<CPUSpecificParams> CPUParams_3D_1x1_I8 = {
        conv_sse42_3D_1x1_nspc,
        conv_avx2_3D_1x1_nspc,
        conv_avx512_3D_1x1_nspc
};

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_1x1_I8, DeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        convParams_ExplicitPadding_1x1_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8, Precision::I8),
                                        ::testing::Values(Precision::FP32/*, Precision::I32*/),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7, 7 })),
                                        ::testing::ValuesIn(emptyOutputShape),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_1x1_I8)),
                                ::testing::ValuesIn(fusingParamsSetI8),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        DeconvolutionLayerCPUTest::getTestCaseName);

/* ========= */

} // namespace
} // namespace CPULayerTestsDefinitions
