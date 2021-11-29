// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/conv_fq_eltwise.hpp"

namespace SubgraphTestsDefinitions {

std::string ConvFqEltwiseTest::getTestCaseName(testing::TestParamInfo<ConvFqEltwiseTestParamsSet> obj) {
    FqSpecificParams fqParams;
    ConvParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(fqParams, convParams, netPrecision, inputShapes, targetDevice, config) = obj.param;

    size_t levels;
    std::vector<float> inputArg;
    std::tie(levels, inputArg) = fqParams;

    std::vector<size_t> kernelShape;
    std::vector<size_t> strides;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(kernelShape, strides, inputChannels, outputChannels) = convParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "LEVELS=" << levels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
     if (inputArg.size() == 3) {
        result << "_inputArg=" << inputArg[0] << "_" << inputArg[1] << "_" << inputArg[2];
    }
    result << "_KERNEL=" << CommonTestUtils::vec2str(kernelShape) << "_";
    result << "STRIDES=" << CommonTestUtils::vec2str(strides) << "_";
    result << "IC=" << inputChannels << "_";
    result << "OC=" << outputChannels;
    return result.str();
}

void ConvFqEltwiseTest::SetUp() {
    FqSpecificParams fqParams;
    ConvParams convParams;
    std::vector<size_t> inputShape;
    std::map<std::string, std::string> config;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(fqParams, convParams, netPrecision, inputShape, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());

    size_t levels;
    std::vector<float> inputArg;
    std::tie(levels, inputArg) = fqParams;
    if (inputArg.size() == 3) {
        inputDataMin = inputArg[0];
        inputDataMax = inputArg[1];
        inputDataResolution = inputArg[2];
    }

    std::vector<size_t> kernelShape;
    std::vector<size_t> strides;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(kernelShape, strides, inputChannels, outputChannels) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    const int seed = 0;
    std::mt19937 gen(seed);

    std::vector<size_t> convInputShape = {1, inputChannels, 1, inputShape[0] * inputShape[1] / inputChannels};
    auto reshapePattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, convInputShape);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], reshapePattern1, false);

    float weightVal = 0.2;
    auto filterWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, {outputChannels, inputChannels, kernelShape[0], kernelShape[1]},
                                                                  { weightVal });
    auto convLowNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 }, std::vector<float>{-weightVal});
    auto convHighNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 }, std::vector<float>{weightVal});
    auto convWeightsFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(filterWeightsNode,
        convLowNode, convHighNode, convLowNode, convHighNode, levels);
    auto convWeightsFQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(convWeightsFQNode);
    auto conv = std::make_shared<ngraph::opset1::Convolution>(reshape1, convWeightsFQ, strides, std::vector<ptrdiff_t>{ 0, 0 },
                                                              std::vector<ptrdiff_t>{ 0, 0 }, std::vector<size_t>{ 1, 1 },
                                                              ngraph::op::PadType::VALID);
    auto biasesWeightsNode = ngraph::builder::makeConstant(ngPrc, {}, std::vector<float>{ 0.0f });
    auto add_1 = std::make_shared<ngraph::opset1::Add>(conv, biasesWeightsNode);

    auto widthAfterConv = (convInputShape[3] - kernelShape[1]) / strides[1] + 1;
    auto heightAfterConv = (convInputShape[2] - kernelShape[0]) / strides[0] + 1;
    std::vector<size_t> outFormShapes = {1,  outputChannels * widthAfterConv * heightAfterConv };

    auto lowNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 },
        std::vector<float>{inputDataMin * weightVal * kernelShape[1] * 1.5f});
    auto highNode = ngraph::builder::makeConstant(ngraph::element::f32, std::vector<size_t>{ 1 },
        std::vector<float>{inputDataMax * weightVal * kernelShape[1] * 1.5f});
    auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(add_1, lowNode, highNode, lowNode, highNode, levels);

    auto constNode = ngraph::builder::makeConstant(ngPrc, {}, std::vector<float>{ 0.5f });
    auto add_2 = std::make_shared<ngraph::opset1::Add>(fq, constNode);

    auto reshapePattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(add_2, reshapePattern2, false);

    function = std::make_shared<ngraph::Function>(reshape2, params, "convFqEltwise");
}

InferenceEngine::Blob::Ptr ConvFqEltwiseTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution,
                                            seed);
}
}  // namespace SubgraphTestsDefinitions