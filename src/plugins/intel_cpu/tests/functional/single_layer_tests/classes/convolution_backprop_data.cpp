// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_backprop_data.hpp"

#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPULayerTestsDefinitions;

std::string DeconvolutionLayerCPUTest::getTestCaseName(testing::TestParamInfo<DeconvLayerCPUTestParamsSet> obj) {
    DeconvSpecParams basicParamsSet;
    DeconvInputData inputData;
    ElementType prec;
    fusingSpecificParams fusingParams;
    CPUSpecificParams cpuParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = obj.param;

    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

    InputShape inputShape;
    ngraph::helpers::InputLayerType outShapeType;
    std::vector<std::vector<int32_t>> outShapeData;
    std::tie(inputShape, outShapeType, outShapeData) = inputData;

    std::ostringstream result;
    result << "IS=";
    result << ov::test::utils::partialShape2str({inputShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShape.second) {
        result << "(";
        result << ov::test::utils::vec2str(shape);
        result << ")_";
    }
    result << "PRC=" << prec << "_";
    result << "K=" << ov::test::utils::vec2str(kernel) << "_";
    result << "S=" << ov::test::utils::vec2str(stride) << "_";
    result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "OP=" << ov::test::utils::vec2str(outPadding) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "OUT_SH=" << outShapeType << "_";
    result << "OUT_D=";
    for (const auto& data : outShapeData) {
        result << "(";
        result << ov::test::utils::vec2str(data);
        result << ")_";
    }

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

void DeconvolutionLayerCPUTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto &funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto &funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i == 1) {
            tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i],
                                outShapeData[inferRequestNum].data());
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i],
                                                             2560, 0, 256);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
    inferRequestNum++;
}

void DeconvolutionLayerCPUTest::init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) {
    if (function->get_parameters().size() == 1) {
        ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
    } else {
        // WA: output_shape depends on 3rd deconvolution input data
        // but the reference implementation doesn't implement shape inference
        // so we need to build a new ngraph function and replace the 3rd input parameter with a constant
        // to get valid output shapes
        funcRef = createGraph({targetInputStaticShapes[0]}, ngraph::helpers::InputLayerType::CONSTANT);
    }
}

void DeconvolutionLayerCPUTest::validate() {
    auto actualOutputs = get_plugin_outputs();
    if (function->get_parameters().size() == 2) {
        auto pos = std::find_if(inputs.begin(), inputs.end(),
                                [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor> &params) {
                                    return params.first->get_friendly_name() == "param_1";
                                });
        IE_ASSERT(pos != inputs.end());
        inputs.erase(pos);
    }
    auto expectedOutputs = calculate_refs();
    if (expectedOutputs.empty()) {
        return;
    }
    ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
                                << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE "
                                << actualOutputs.size();

    compare(expectedOutputs, actualOutputs);
}

void DeconvolutionLayerCPUTest::configure_model() {
    ov::preprocess::PrePostProcessor p(function);
    {
        auto &params = function->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            if (i > 0) {
                continue;
            }
            if (inType != ov::element::Type_t::undefined) {
                p.input(i).tensor().set_element_type(inType);
            }
        }
    }
    {
        auto results = function->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            if (outType != ov::element::Type_t::undefined) {
                p.output(i).tensor().set_element_type(outType);
            }
        }
    }
    function = p.build();
}

std::shared_ptr<ov::Model> DeconvolutionLayerCPUTest::createGraph(const std::vector<ov::PartialShape>& inShapes, ngraph::helpers::InputLayerType outShapeType) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(prec, inShapes.front())};
    std::shared_ptr<ov::Node> outShapeNode;
    if (!outShapeData.empty()) {
        if (outShapeType == ngraph::helpers::InputLayerType::PARAMETER) {
            IE_ASSERT(inputDynamicShapes.size() == 2);
            auto outShapeParam = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, inputDynamicShapes.back());
            params.push_back(outShapeParam);
            outShapeNode = outShapeParam;
        } else {
            outShapeNode = ngraph::opset8::Constant::create(ngraph::element::i32, {outShapeData[inferRequestNum].size()}, outShapeData[inferRequestNum]);
        }
    }

    for (size_t i = 0; i < params.size(); i++) {
        params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
    }

    std::shared_ptr<ov::Node> deconv;
    if (!outShapeData.empty()) {
        IE_ASSERT(outShapeNode != nullptr);
        deconv = ngraph::builder::makeConvolutionBackpropData(params[0], outShapeNode, prec, kernel, stride, padBegin,
                                                              padEnd, dilation, padType, convOutChannels);
    } else {
        deconv = ngraph::builder::makeConvolutionBackpropData(params[0], prec, kernel, stride, padBegin,
                                                              padEnd, dilation, padType, convOutChannels, false, outPadding);
    }

    return makeNgraphFunction(prec, params, deconv, "DeconvCPU");
}

void DeconvolutionLayerCPUTest::SetUp() {
    rel_threshold = 1e-4f;

    targetDevice = ov::test::utils::DEVICE_CPU;

    DeconvSpecParams basicParamsSet;
    DeconvInputData inputData;
    fusingSpecificParams fusingParams;
    CPUSpecificParams cpuParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = this->GetParam();

    InputShape inputShape;
    ngraph::helpers::InputLayerType outShapeType;
    std::tie(inputShape, outShapeType, outShapeData) = inputData;

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

    if (additionalConfig[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] ==
        InferenceEngine::PluginConfigParams::YES) {
        inType = outType = prec = ElementType::bf16;
        rel_threshold = 1e-2f;
    } else {
        inType = outType = prec;
    }

    selectedType = makeSelectedTypeStr(selectedType, prec);

    std::vector<InputShape> paramsShapes;
    paramsShapes.push_back(inputShape);
    if (!outShapeData.empty() && outShapeType == ngraph::helpers::InputLayerType::PARAMETER) {
        const auto outShapeDims = ov::Shape{outShapeData.front().size()};
        paramsShapes.push_back(
                InputShape{outShapeDims, std::vector<ov::Shape>(inputShape.second.size(), outShapeDims)});
    }

    init_input_shapes(paramsShapes);

    function = createGraph(inputDynamicShapes, outShapeType);
}

TEST_P(DeconvolutionLayerCPUTest, CompareWithRefs) {
    if (!fusedOps.empty()) {
        bool isSupportedParams = stride[stride.size() - 1] <= kernel[kernel.size() - 1];
        if (stride.size() > 1)
            isSupportedParams &= stride[stride.size() - 2] <= kernel[kernel.size() - 2];
        if (stride.size() > 2)
            isSupportedParams &= stride[stride.size() - 3] <= kernel[kernel.size() - 3];
        if (!isSupportedParams) {
            GTEST_SKIP() << "Fusing with strides more than kernel size was disabled, because oneDNN deconvolution doesn't support it"
                         << std::endl;
        }
    }

    run();
    CheckPluginRelatedResults(compiledModel, "Deconvolution");
}
