// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution.hpp"

#include "gtest/gtest.h"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace CPUTestUtils;
using namespace ov::intel_cpu;

namespace ov {
namespace test {
namespace Convolution {

std::string ConvolutionLayerCPUTest::getTestCaseName(const testing::TestParamInfo<convLayerCPUTestParamsSet>& obj) {
    convLayerTestParamsSet basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

    convSpecificParams convParams;
    ElementType netType;
    ElementType inType, outType;
    InputShape inputShape;
    std::string targetDevice;
    std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = basicParamsSet;
    ov::op::PadType padType;
    ov::Shape kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=";
    result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    result << "TS=(";
    for (const auto& shape : inputShape.second) {
        result << ov::test::utils::vec2str(shape) << "_";
    }
    result << ")_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netType << "_";
    result << "inPRC=" << inType << "_";
    result << "outPRC=" << outType << "_";
    result << "trgDev=" << targetDevice;

    result << CPUTestsBase::getTestCaseName(cpuParams);
    result << CpuTestWithFusing::getTestCaseName(fusingParams);

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }

    return result.str();
}

void ConvolutionLayerCPUTest::checkBiasFusing(ov::CompiledModel& execNet) const {
    if (!execNet)
        return;

    auto execGraph = execNet.get_runtime_model();
    ASSERT_NE(nullptr, execGraph);

    bool foundConv = false;
    for (const auto& node : execGraph->get_ops()) {
        const auto& rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        if (getExecValue(ov::exec_model_info::LAYER_TYPE) == "Convolution") {
            foundConv = true;
            ASSERT_EQ(3, node->inputs().size());
            break;
        }
    }

    ASSERT_TRUE(foundConv) << "Can't find Convolution node";
}

std::shared_ptr<ov::Node> ConvolutionLayerCPUTest::modifyGraph(const ov::element::Type& ngPrc,
                                                               ov::ParameterVector& params,
                                                               const std::shared_ptr<ov::Node>& lastNode) {
    auto retNode = CpuTestWithFusing::modifyGraph(ngPrc, params, lastNode);
    std::shared_ptr<ov::Node> opToShapeInfer = nullptr;
    for (auto& targetShapes : targetStaticShapes) {
        for (size_t i = targetShapes.size(); i < params.size(); ++i) {
            const auto& shape = params[i]->get_output_partial_shape(0);
            if (shape.is_static()) {
                targetShapes.push_back(shape.get_shape());
            } else {
                // It is assumed that in such tests we have second parameter only if sum fusion is tested.
                // Considering this fact, we need to set the appropriate static shape for the second term of the sum
                // operation, and it has to match the convolution output shape. So the most suitable solution here is to
                // perform shape inference on the convolution node
                if (!opToShapeInfer) {
                    ov::OutputVector inputsForShapeInfer;
                    for (size_t j = 0; j < lastNode->get_input_size(); j++) {
                        if (ov::is_type<ov::op::v0::Constant>(lastNode->get_input_node_ptr(j))) {
                            inputsForShapeInfer.push_back(lastNode->get_input_node_shared_ptr(j));
                        } else {
                            inputsForShapeInfer.push_back(
                                std::make_shared<ov::op::v0::Parameter>(lastNode->get_input_element_type(j),
                                                                        lastNode->get_input_partial_shape(j)));
                        }
                    }
                    opToShapeInfer = lastNode->clone_with_new_inputs(inputsForShapeInfer);
                }

                std::vector<ov::Shape> secondParameterShapes;
                if (auto parameter = dynamic_cast<ov::op::v0::Parameter*>(opToShapeInfer->get_input_node_ptr(0))) {
                    parameter->set_partial_shape(targetShapes.front());
                    parameter->validate_and_infer_types();
                }
                opToShapeInfer->validate_and_infer_types();
                targetShapes.push_back(opToShapeInfer->get_output_shape(0));
            }
        }
    }
    return retNode;
}

void ConvolutionLayerCPUTest::SetUp() {
    rel_threshold = 1e-4f;

    convLayerTestParamsSet basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = this->GetParam();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    if (postOpMgrPtr)
        isBias = (postOpMgrPtr->getFusedOpsNames() == "Add(PerChannel)" && selectedType != "jit_avx512_winograd");

    convSpecificParams convParams;
    InputShape inputShape;
    auto netType = ElementType::undefined;
    std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = basicParamsSet;

    init_input_shapes({inputShape});

    auto it = configuration.find(ov::hint::inference_precision.name());
    ov::element::Type inference_precision = (it != configuration.end()) ?
                                            it->second.as<ov::element::Type>() : ov::element::undefined;
    if (inference_precision == ov::element::bf16) {
        selectedType += "_BF16";
        rel_threshold = 1e-2f;
        if (selectedType == "jit_gemm_BF16")
            rel_threshold = 0.05f;
    } else if (inference_precision == ov::element::f16) {
            selectedType +=  "_FP16";
            rel_threshold = 0.00125f;
    } else {
        selectedType = makeSelectedTypeStr(selectedType, netType);
    }

    ov::op::PadType padType;
    ov::Shape stride;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    ov::ParameterVector inputParams;
    for (auto&& shape : inputDynamicShapes)
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
    auto convolutionNode = ov::test::utils::make_convolution(inputParams[0], netType, kernel, stride, padBegin,
                                                            padEnd, dilation, padType, convOutChannels);

    function = makeNgraphFunction(netType, inputParams, convolutionNode, "Convolution");
}

TEST_P(ConvolutionLayerCPUTest, CompareWithRefs) {
    // Skip tests for sse41 convolution where ic or oc cannot be exactly divided by the block size,
    // since tails processing for sse41 nspc layout is not supported yet (see 52736).
    if (!inFmts.empty() && (inFmts.front() == nwc || inFmts.front() == nhwc || inFmts.front() == ndhwc) && selectedType.find("jit_sse") != std::string::npos) {
        auto inpChannels = function->get_parameters().front()->get_partial_shape()[1].get_length();
        auto outChannels = function->get_output_partial_shape(0)[1].get_length();
        if ((inpChannels % 8) || (outChannels % 8)) {
            GTEST_SKIP() << "Disabled test due to the sse41 convolution kernel does not support tails for nspc layout." << std::endl;
        }
    }

    if (!priority.empty()) {
        // Skip tests for brgconv convolution where kernel size = 1x1
        if (one_of(priority[0], "brgconv_avx512", "brgconv_avx512_amx", "brgconv_avx2")) {
                bool is_1x1 = true;
                for (const auto &i : kernel) {
                if (i != 1) {
                        is_1x1 = false;
                        break;
                }
                }
                if (is_1x1) {
                GTEST_SKIP() << "Disabled test due to the brgconv does not support 1x1 convolution kernel." << std::endl;
                }
        }

        // Skip tests for brgconv_amx convolution where dilation is not 1
        if (priority[0].find("amx") != std::string::npos) {
                bool dilation_is_1x1 = true;
                for (const auto &i : dilation) {
                if (i != 1) {
                        dilation_is_1x1 = false;
                        break;
                }
                }
                if (!dilation_is_1x1) {
                GTEST_SKIP() << "Disabled test due to the brgconv amx does not support non 1 dilation convolution kernel." << std::endl;
                }
        }
    }

// FIXME: ACL output shape check fails if kernel, stride and padding equal to 1
// CpuGemm::validate checks that 2nd and 3rd dimention of the input and output shapes are equal and fails (ticket 114201)
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if (std::all_of(kernel.begin(), kernel.end(), [](size_t i){return i == 1;}) &&
        std::all_of(stride.begin(), stride.end(), [](size_t i){return i == 1;}) &&
        std::all_of(padBegin.begin(), padBegin.end(), [](ptrdiff_t i){return i == 1;})) {
        GTEST_SKIP() << "Disabled test due to output shape check failed" << std::endl;
    }
#endif
    run();

    if (isBias) {
        checkBiasFusing(compiledModel);
    }
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

const ov::Shape& numOutChannels() {
    static const ov::Shape numOutChannels = { 64, 63 };
    return numOutChannels;
}

const ov::Shape& numOutChannels_Gemm() {
    static const ov::Shape numOutChannels_Gemm = { 6 };
    return numOutChannels_Gemm;
}

const std::vector<SizeVector>& kernels1d() {
    static const std::vector<SizeVector> kernels1d = { {3}, {1} };
    return kernels1d;
}

const std::vector<SizeVector>& strides1d() {
    static const std::vector<SizeVector> strides1d = { {1}, {2} };
    return strides1d;
}

const std::vector<std::vector<ptrdiff_t>>& padBegins1d() {
    static const std::vector<std::vector<ptrdiff_t>> padBegins1d = { {0}, {1} };
    return padBegins1d;
}

const std::vector<std::vector<ptrdiff_t>>& padEnds1d() {
    static const std::vector<std::vector<ptrdiff_t>> padEnds1d = { {0} };
    return padEnds1d;
}

const std::vector<SizeVector>& dilations1d() {
    static const std::vector<SizeVector> dilations1d = { {1}, {2} };
    return dilations1d;
}

const std::vector<SizeVector>& kernels2d() {
    static const std::vector<SizeVector> kernels2d = { {3, 3}, {1, 1} };
    return kernels2d;
}

const std::vector<SizeVector>& strides2d() {
    static const std::vector<SizeVector> strides2d = { {1, 1}, {2, 2} };
    return strides2d;
}

const std::vector<std::vector<ptrdiff_t>>& padBegins2d() {
    static const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0}, {1, 1} };
    return padBegins2d;
}

const std::vector<std::vector<ptrdiff_t>>& padEnds2d() {
    static const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
    return padEnds2d;
}

const std::vector<SizeVector>& dilations2d() {
    static const std::vector<SizeVector> dilations2d = { {1, 1} };
    return dilations2d;
}

const std::vector<InputShape>& inShapesGemm2D() {
    static const std::vector<InputShape> inShapesGemm2D = {
            {{}, {{ 2, 12, 7, 7 }}},
            {
                //dynamic shape
                { {1, 200}, 12, -1, {1, 200} },
                { //target static shapes
                    { 2, 12, 7, 7 },
                    { 1, 12, 5, 5 }
                }
            }
    };
    return inShapesGemm2D;
}

const std::vector<InputShape>& inShapesGemm2D_cache() {
    static const std::vector<InputShape> inShapesGemm2D_cache = {
            {{}, {{ 2, 12, 7, 7 }}},
            {
                //dynamic shape
                { {1, 200}, 12, -1, {1, 200} },
                { //target static shapes
                    { 1, 12, 5, 5 },
                    { 1, 12, 7, 7 },
                    { 1, 12, 5, 5 }
                }
            }
    };
    return inShapesGemm2D_cache;
}

const std::vector<CPUSpecificParams>& CPUParams_2D() {
    static const std::vector<CPUSpecificParams> CPUParams_2D = {
        conv_sse42_2D,
        conv_avx2_2D,
        conv_avx512_2D,
        conv_sse42_2D_nspc,
        conv_avx2_2D_nspc,
        conv_avx2_2D_nspc_brgconv,
        conv_avx512_2D_nspc,
        conv_avx512_2D_nspc_brgconv
    };
    return CPUParams_2D;
}

const std::vector<CPUSpecificParams>& CPUParams_3D() {
    static const std::vector<CPUSpecificParams> CPUParams_3D = {
        //conv_sse42_3D, // not supported jit_sse42 for 3d
        conv_avx2_3D,
        conv_avx512_3D,
        conv_avx2_3D_nspc,
        conv_avx2_3D_nspc_brgconv,
        conv_avx512_3D_nspc,
        conv_avx512_3D_nspc_brgconv
    };
    return CPUParams_3D;
}

const std::vector<CPUSpecificParams>& CPUParams_GEMM_1D() {
    static const std::vector<CPUSpecificParams> CPUParams_GEMM_1D = {
            conv_gemm_1D,
            conv_gemm_1D_nspc
    };
    return CPUParams_GEMM_1D;
}

const std::vector<CPUSpecificParams>& CPUParams_GEMM_2D() {
    static const std::vector<CPUSpecificParams> CPUParams_GEMM_2D = {
        conv_gemm_2D,
        conv_gemm_2D_nspc,
        conv_gemm_acl_2D_nspc
    };
    return CPUParams_GEMM_2D;
}

const std::vector<InputShape>& inputShapes1d() {
    static const std::vector<InputShape> inputShapes1d = {
            {{}, {{ 2, 64, 7 }}},
            {{}, {{ 1, 67, 7 }}},
            {
                //dynamic shape
                { -1, 64, {1, 200} },
                { //target static shapes
                    { 2, 64, 7 },
                    { 1, 64, 9 }
                }
            },
            {
                //dynamic shape
                { -1, 67, {1, 200} },
                { //target static shapes
                    { 2, 67, 7 },
                    { 1, 67, 9 }
                }
            },
            {
                //dynamic shape
                { {1, 200}, 64, -1 },
                { //target static shapes
                    { 2, 64, 7 },
                    { 1, 64, 5 }
                }
            }
    };
    return inputShapes1d;
}

const std::vector<InputShape>& inputShapes2d() {
    static const std::vector<InputShape> inputShapes2d = {
            {{}, {{ 1, 64, 7, 7 }}},
            {{}, {{ 1, 67, 7, 7 }}},
            {
                //dynamic shape
                { -1, 64, -1, {1, 200} },
                { //target static shapes
                    { 2, 64, 7, 7 },
                    { 1, 64, 9, 9}
                }
            },
            {
                //dynamic shape
                { -1, 67, -1, {1, 200} },
                { //target static shapes
                    { 2, 67, 7, 7 },
                    { 1, 67, 9, 9}
                }
            }
    };
    return inputShapes2d;
}

const std::vector<InputShape>& inputShapesPlain2Blocked2d() {
    static const std::vector<InputShape> inputShapesPlain2Blocked2d = {
            {{}, {{ 1, 1, 7, 7 }}},
            {{}, {{ 1, 2, 7, 7 }}},
            {{}, {{ 1, 3, 7, 7 }}},
            {
                //dynamic shape
                { -1, 1, -1, {1, 200} },
                { //target static shapes
                    { 2, 1, 7, 7 },
                    { 1, 1, 9, 9}
                }
            },
            {
                //dynamic shape
                { -1, 3, -1, {1, 200} },
                { //target static shapes
                    { 2, 3, 7, 7 },
                    { 1, 3, 9, 9}
                }
            }
    };
    return inputShapesPlain2Blocked2d;
}

const std::vector<InputShape>& inputShapes2d_dynBatch() {
    static const std::vector<InputShape> inputShapes2d_dynBatch = {
            {
                //dynamic shape
                { {1, 10}, 64, 7, 7 },
                { //target static shapes
                    { 2, 64, 7, 7 },
                    { 1, 64, 7, 7 }
                }
            },
    };
    return inputShapes2d_dynBatch;
}

const std::vector<CPUSpecificParams>& CPUParams_1x1_1D() {
    static const std::vector<CPUSpecificParams> CPUParams_1x1_1D = {
            conv_sse42_1D_1x1,
            conv_avx2_1D_1x1,
            conv_avx512_1D_1x1,
            conv_sse42_1D_1x1_nspc,
            conv_avx2_1D_1x1_nspc,
            conv_avx2_1D_1x1_nspc_brgconv,
            conv_avx512_1D_1x1_nspc,
            conv_avx512_1D_1x1_nspc_brgconv
    };
    return CPUParams_1x1_1D;
}

const std::vector<SizeVector>& kernels3d() {
    static const std::vector<SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
    return kernels3d;
}

const std::vector<SizeVector>& strides3d() {
    static const std::vector<SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
    return strides3d;
}

const std::vector<std::vector<ptrdiff_t>>& padBegins3d() {
    static const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0}, {1, 1, 1} };
    return padBegins3d;
}

const std::vector<std::vector<ptrdiff_t>>& padEnds3d() {
    static const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
    return padEnds3d;
}

const std::vector<SizeVector>& dilations3d() {
    static const std::vector<SizeVector> dilations3d = { {1, 1, 1} };
    return dilations3d;
}

const std::vector<InputShape> & inputShapes3d() {
    static const std::vector<InputShape> inputShapes3d = {
            {{}, {{ 1, 64, 7, 7, 7 }}},
            {{}, {{ 1, 67, 7, 7, 7 }}},
            {
                //dynamic shapes
                { -1, 64, -1, {1, 200}, -1 },
                { //target static shapes
                    { 1, 64, 7, 7, 7 },
                    { 1, 64, 9, 9, 9}
                }
            },
            {
                //dynamic shapes
                { -1, 67, -1, {1, 200}, -1 },
                { //target static shapes
                    { 1, 67, 7, 7, 7 },
                    { 1, 67, 9, 9, 9}
                }
            }
    };
    return inputShapes3d;
}

const std::vector<InputShape> & inShapesGemm3D() {
    static const std::vector<InputShape> inShapesGemm3D = {
            {{}, {{ 2, 12, 7, 7, 7 }}},
            {
                //dynamic shape
                { {1, 200}, 12, -1, {1, 200}, -1 },
                { //target static shapes
                    { 2, 12, 7, 7, 7 },
                    { 1, 12, 5, 5, 5 }
                }
            }
    };
    return inShapesGemm3D;
}

const std::vector<CPUSpecificParams>& CPUParams_GEMM_3D() {
    static const std::vector<CPUSpecificParams> CPUParams_GEMM_3D = {
            conv_gemm_3D,
            conv_gemm_3D_nspc,
            conv_gemm_acl_3D,
            conv_gemm_acl_3D_nspc
    };
    return CPUParams_GEMM_3D;
}

const std::vector<CPUSpecificParams>& CPUParams_1x1_2D() {
    static const std::vector<CPUSpecificParams> CPUParams_1x1_2D = {
            conv_sse42_2D_1x1,
            conv_avx2_2D_1x1,
            conv_avx512_2D_1x1,
            conv_sse42_2D_1x1_nspc,
            conv_avx2_2D_1x1_nspc,
            conv_avx2_2D_1x1_nspc_brgconv,
            conv_avx512_2D_1x1_nspc,
            conv_avx512_2D_1x1_nspc_brgconv
    };
    return CPUParams_1x1_2D;
}

const std::vector<InputShape>& inputShapes2d_cache() {
    static const std::vector<InputShape> inputShapes2d_cache = {
            {{}, {{ 1, 64, 7, 7 }}},
            {{}, {{ 1, 67, 7, 7 }}},
            {
                //dynamic shape
                { -1, 64, -1, {1, 200} },
                { //target static shapes
                    { 1, 64, 7, 7 },
                    { 1, 64, 9, 9 },
                    { 1, 64, 7, 7 }
                }
            },
            {
                //dynamic shape
                { -1, 67, -1, {1, 200} },
                { //target static shapes
                    { 1, 67, 7, 7 },
                    { 1, 67, 9, 9}
                }
            }
    };
    return inputShapes2d_cache;
}

const std::vector<fusingSpecificParams>& fusingParamsSetWithEmpty() {
    static const std::vector<fusingSpecificParams> fusingParamsSetWithEmpty = {
            emptyFusingSpec,
            // eltwise
            fusingRelu,
            fusingPRelu1DScaleShift,
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
    return fusingParamsSetWithEmpty;
}

const std::vector<InputShape>& inShapesGemm1D() {
    static const std::vector<InputShape> inShapesGemm1D = {
            {{}, {{ 2, 12, 7 }}},
            {
                //dynamic shape
                { {1, 200}, 12, {1, 200} },
                { //target static shapes
                    { 2, 12, 7 },
                    { 1, 12, 5 }
                }
            }
    };
    return inShapesGemm1D;
}

const convParams_ExplicitPaddingType& convParams_ExplicitPadding_GEMM_2D() {
    static const auto convParams_ExplicitPadding_GEMM_2D =
        ::testing::Combine(::testing::ValuesIn(kernels2d()),
                           ::testing::ValuesIn(strides2d()),
                           ::testing::ValuesIn(padBegins2d()),
                           ::testing::ValuesIn(padEnds2d()),
                           ::testing::ValuesIn(dilations2d()),
                           ::testing::ValuesIn(numOutChannels_Gemm()),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_GEMM_2D;
}

const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_GEMM_2D_dilated() {
    static const auto convParams_ExplicitPadding_GEMM_2D_dilated =
        ::testing::Combine(::testing::ValuesIn(kernels2d()),
                           ::testing::ValuesIn(strides2d()),
                           ::testing::ValuesIn(padBegins2d()),
                           ::testing::ValuesIn(padEnds2d()),
                           ::testing::Values(ov::Shape{2, 2}),
                           ::testing::ValuesIn(numOutChannels_Gemm()),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_GEMM_2D_dilated;
}

const convParams_ExplicitPaddingType& convParams_ExplicitPadding_GEMM_1D() {
    static const auto convParams_ExplicitPadding_GEMM_1D =
        ::testing::Combine(::testing::ValuesIn(kernels1d()),
                           ::testing::ValuesIn(strides1d()),
                           ::testing::ValuesIn(padBegins1d()),
                           ::testing::ValuesIn(padEnds1d()),
                           ::testing::ValuesIn(dilations1d()),
                           ::testing::ValuesIn(numOutChannels_Gemm()),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_GEMM_1D;
}

const convParams_ExplicitPaddingType& convParams_ExplicitPadding_2D() {
    static const auto convParams_ExplicitPadding_2D = ::testing::Combine(::testing::ValuesIn(kernels2d()),
                                                                         ::testing::ValuesIn(strides2d()),
                                                                         ::testing::ValuesIn(padBegins2d()),
                                                                         ::testing::ValuesIn(padEnds2d()),
                                                                         ::testing::ValuesIn(dilations2d()),
                                                                         ::testing::ValuesIn(numOutChannels()),
                                                                         ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_2D;
}

const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_2D_dilated() {
    static const auto convParams_ExplicitPadding_2D_dilated =
        ::testing::Combine(::testing::ValuesIn(kernels2d()),
                           ::testing::ValuesIn(strides2d()),
                           ::testing::ValuesIn(padBegins2d()),
                           ::testing::ValuesIn(padEnds2d()),
                           ::testing::Values(ov::Shape{2, 2}),
                           ::testing::ValuesIn(numOutChannels()),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_2D_dilated;
}

const convParams_ExplicitPaddingType& convParams_ExplicitPadding_GEMM_3D() {
    static const auto convParams_ExplicitPadding_GEMM_3D =
        ::testing::Combine(::testing::ValuesIn(kernels3d()),
                           ::testing::ValuesIn(strides3d()),
                           ::testing::ValuesIn(padBegins3d()),
                           ::testing::ValuesIn(padEnds3d()),
                           ::testing::ValuesIn(dilations3d()),
                           ::testing::ValuesIn(numOutChannels_Gemm()),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_GEMM_3D;
}

const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_GEMM_3D_dilated() {
    static const auto convParams_ExplicitPadding_GEMM_3D_dilated =
        ::testing::Combine(::testing::ValuesIn(kernels3d()),
                           ::testing::ValuesIn(strides3d()),
                           ::testing::ValuesIn(padBegins3d()),
                           ::testing::ValuesIn(padEnds3d()),
                           ::testing::Values(ov::Shape{2, 2, 2}),
                           ::testing::ValuesIn(numOutChannels_Gemm()),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_GEMM_3D_dilated;
}

const convParams_ExplicitPaddingType& convParams_ExplicitPadding_3D() {
    static const auto convParams_ExplicitPadding_3D = ::testing::Combine(::testing::ValuesIn(kernels3d()),
                                                                         ::testing::ValuesIn(strides3d()),
                                                                         ::testing::ValuesIn(padBegins3d()),
                                                                         ::testing::ValuesIn(padEnds3d()),
                                                                         ::testing::ValuesIn(dilations3d()),
                                                                         ::testing::ValuesIn(numOutChannels()),
                                                                         ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_3D;
}

const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_3D_dilated() {
    static const auto convParams_ExplicitPadding_3D_dilated =
        ::testing::Combine(::testing::ValuesIn(kernels3d()),
                           ::testing::ValuesIn(strides3d()),
                           ::testing::ValuesIn(padBegins3d()),
                           ::testing::ValuesIn(padEnds3d()),
                           ::testing::Values(ov::Shape{2, 2, 2}),
                           ::testing::ValuesIn(numOutChannels()),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_3D_dilated;
}

const convParams_ExplicitPadding_1x1_Type& convParams_ExplicitPadding_1x1_1D() {
    static const auto convParams_ExplicitPadding_1x1_1D =
        ::testing::Combine(::testing::Values(ov::Shape({1})),
                           ::testing::Values(ov::Shape({1})),
                           ::testing::Values(std::vector<ptrdiff_t>({0})),
                           ::testing::Values(std::vector<ptrdiff_t>({0})),
                           ::testing::Values(ov::Shape({1})),
                           ::testing::Values(63),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_1x1_1D;
}

const convParams_ExplicitPadding_1x1_Type& convParams_ExplicitPadding_1x1_2D() {
    static const auto convParams_ExplicitPadding_1x1_2D =
        ::testing::Combine(::testing::Values(ov::Shape({1, 1})),
                           ::testing::Values(ov::Shape({1, 1})),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                           ::testing::Values(ov::Shape({1, 1})),
                           ::testing::Values(63),
                           ::testing::Values(ov::op::PadType::EXPLICIT));
    return convParams_ExplicitPadding_1x1_2D;
}

}  // namespace Convolution
}  // namespace test
}  // namespace ov
