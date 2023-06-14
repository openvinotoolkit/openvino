// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution.hpp"

#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
std::string ConvolutionLayerCPUTest::getTestCaseName(const testing::TestParamInfo<convLayerCPUTestParamsSet>& obj) {
    convLayerTestParamsSet basicParamsSet;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(basicParamsSet, cpuParams, fusingParams, additionalConfig) = obj.param;

    convSpecificParams convParams;
    ElementType netType;
    ElementType inType, outType;
    InputShape inputShape;
    std::string targetDevice;
    std::tie(convParams, netType, inType, outType, inputShape, targetDevice) = basicParamsSet;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=";
    result << CommonTestUtils::partialShape2str({inputShape.first}) << "_";
    result << "TS=(";
    for (const auto& shape : inputShape.second) {
        result << CommonTestUtils::vec2str(shape) << "_";
    }
    result << ")_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
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
            result << "_" << item.first << "=" << item.second;
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
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Convolution") {
            foundConv = true;
            ASSERT_EQ(3, node->inputs().size());
            break;
        }
    }

    ASSERT_TRUE(foundConv) << "Can't find Convolution node";
}

std::shared_ptr<ngraph::Node> ConvolutionLayerCPUTest::modifyGraph(const ngraph::element::Type& ngPrc,
                                                                   ngraph::ParameterVector& params,
                                                                   const std::shared_ptr<ngraph::Node>& lastNode) {
    auto retNode = CpuTestWithFusing::modifyGraph(ngPrc, params, lastNode);
    std::shared_ptr<ngraph::Node> opToShapeInfer = nullptr;
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
                    ngraph::OutputVector inputsForShapeInfer;
                    for (size_t j = 0; j < lastNode->get_input_size(); j++) {
                        if (ngraph::is_type<ngraph::opset1::Constant>(lastNode->get_input_node_ptr(j))) {
                            inputsForShapeInfer.push_back(lastNode->get_input_node_shared_ptr(j));
                        } else {
                            inputsForShapeInfer.push_back(
                                std::make_shared<ngraph::opset1::Parameter>(lastNode->get_input_element_type(j),
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
    std::map<std::string, std::string> additionalConfig;
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

    if (configuration.count(PluginConfigParams::KEY_ENFORCE_BF16) &&
        PluginConfigParams::YES == configuration[PluginConfigParams::KEY_ENFORCE_BF16].as<std::string>()) {
        selectedType += "_BF16";
        rel_threshold = 1e-2f;
        if (selectedType == "jit_gemm_BF16")
            rel_threshold = 0.05f;
    } else {
        selectedType = makeSelectedTypeStr(selectedType, netType);
    }

    ngraph::op::PadType padType;
    InferenceEngine::SizeVector stride;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    auto inputParams = ngraph::builder::makeDynamicParams(ngraph::element::f32, inputDynamicShapes);
    auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

    auto convolutionNode = ngraph::builder::makeConvolution(paramOuts.front(),
                                                            netType,
                                                            kernel,
                                                            stride,
                                                            padBegin,
                                                            padEnd,
                                                            dilation,
                                                            padType,
                                                            convOutChannels);

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
        if (priority[0] == "brgconv_avx512" || priority[0] == "brgconv_avx512_amx") {
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

    run();

    if (isBias) {
        checkBiasFusing(compiledModel);
    }
    CheckPluginRelatedResults(compiledModel, "Convolution");
}
}  // namespace CPULayerTestsDefinitions