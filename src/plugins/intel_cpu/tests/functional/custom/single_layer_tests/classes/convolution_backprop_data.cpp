// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_backprop_data.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

std::string DeconvolutionLayerCPUTest::getTestCaseName(testing::TestParamInfo<DeconvLayerCPUTestParamsSet> obj) {
    DeconvSpecParams basicParamsSet;
    DeconvInputData inputData;
    ElementType prec;
    fusingSpecificParams fusingParams;
    CPUSpecificParams cpuParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = obj.param;

    ov::op::PadType padType;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

    InputShape inputShape;
    ov::test::utils::InputLayerType outShapeType;
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
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }

    return result.str();
}

void DeconvolutionLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    if (function->get_parameters().size() != 1) {
        // WA: output_shape depends on 3rd deconvolution input data
        // but the reference implementation doesn't implement shape inference
        // so we need to build a new function and replace the 3rd input parameter with a constant
        // to get valid output shapes
        functionRefs = createGraph({targetInputStaticShapes[0]}, ov::test::utils::InputLayerType::CONSTANT);
    }
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;

        if (i == 1) {
            tensor = ov::Tensor(funcInput.get_element_type(),
                                targetInputStaticShapes[i],
                                outShapeData[inferRequestNum].data());
        } else {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 2560;
            in_data.resolution = 256;
            tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                             targetInputStaticShapes[i],
                                                             in_data);
        }

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
    inferRequestNum++;
}

void DeconvolutionLayerCPUTest::configure_model() {
    ov::preprocess::PrePostProcessor p(function);
    {
        auto& params = function->get_parameters();
        for (size_t i = 0; i < params.size(); i++) {
            if (i > 0) {
                continue;
            }
            if (inType != ov::element::Type_t::dynamic) {
                p.input(i).tensor().set_element_type(inType);
            }
        }
    }
    {
        auto results = function->get_results();
        for (size_t i = 0; i < results.size(); i++) {
            if (outType != ov::element::Type_t::dynamic) {
                p.output(i).tensor().set_element_type(outType);
            }
        }
    }
    function = p.build();
}

std::shared_ptr<ov::Model> DeconvolutionLayerCPUTest::createGraph(const std::vector<ov::PartialShape>& inShapes, ov::test::utils::InputLayerType outShapeType) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(prec, inShapes.front())};
    std::shared_ptr<ov::Node> outShapeNode;
    if (!outShapeData.empty()) {
        if (outShapeType == ov::test::utils::InputLayerType::PARAMETER) {
            OPENVINO_ASSERT(inputDynamicShapes.size() == 2);
            auto outShapeParam =
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes.back());
            params.push_back(outShapeParam);
            outShapeNode = outShapeParam;
        } else {
            outShapeNode = ov::op::v0::Constant::create(ov::element::i32,
                                                        {outShapeData[inferRequestNum].size()},
                                                        outShapeData[inferRequestNum]);
        }
    }

    for (size_t i = 0; i < params.size(); i++) {
        params[i]->set_friendly_name(std::string("param_") + std::to_string(i));
    }

    std::shared_ptr<ov::Node> deconv;
    if (!outShapeData.empty()) {
        OPENVINO_ASSERT(outShapeNode != nullptr);
        deconv = ov::test::utils::make_convolution_backprop_data(params[0],
                                                                 outShapeNode,
                                                                 prec,
                                                                 kernel,
                                                                 stride,
                                                                 padBegin,
                                                                 padEnd,
                                                                 dilation,
                                                                 padType,
                                                                 convOutChannels);
    } else {
        deconv = ov::test::utils::make_convolution_backprop_data(params[0],
                                                                 prec,
                                                                 kernel,
                                                                 stride,
                                                                 padBegin,
                                                                 padEnd,
                                                                 dilation,
                                                                 padType,
                                                                 convOutChannels,
                                                                 false,
                                                                 outPadding);
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
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputData, prec, fusingParams, cpuParams, additionalConfig) = this->GetParam();

    InputShape inputShape;
    ov::test::utils::InputLayerType outShapeType;
    std::tie(inputShape, outShapeType, outShapeData) = inputData;

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = basicParamsSet;

    auto it = configuration.find(ov::hint::inference_precision.name());
    ov::element::Type inference_precision =
        (it != configuration.end()) ? it->second.as<ov::element::Type>() : ov::element::dynamic;
    if (inference_precision == ov::element::bf16) {
        inType = outType = prec = ElementType::bf16;
        rel_threshold = 1e-2f;
    } else if (inference_precision == ov::element::f16) {
        inType = outType = prec = ElementType::f16;
        rel_threshold = 0.00125f;
    } else {
        inType = outType = prec;
    }

    selectedType = makeSelectedTypeStr(selectedType, prec);

    std::vector<InputShape> paramsShapes;
    paramsShapes.push_back(inputShape);
    if (!outShapeData.empty() && outShapeType == ov::test::utils::InputLayerType::PARAMETER) {
        const auto outShapeDims = ov::Shape{outShapeData.front().size()};
        paramsShapes.push_back(
                InputShape{outShapeDims, std::vector<ov::Shape>(inputShape.second.size(), outShapeDims)});
    }

    init_input_shapes(paramsShapes);

    function = createGraph(inputDynamicShapes, outShapeType);
}
