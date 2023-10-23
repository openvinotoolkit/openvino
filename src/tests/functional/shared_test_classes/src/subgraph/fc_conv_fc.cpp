// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/fc_conv_fc.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string FcAfterConvTest::getTestCaseName(const testing::TestParamInfo<FcConvFcParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t inputChannels;
    size_t outputChannels;
    convParams convolutionParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(netPrecision, targetDevice, configuration, convolutionParams, inputChannels, outputChannels) = obj.param;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "KS=" << ov::test::utils::vec2str(kernelShape) << "_";
    result << "S=" << stride << "_";
    result << "IC=" << inputChannels << "_";
    result << "OC=" << outputChannels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr FcAfterConvTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -2.0f, 2.0f);
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = values[i];
    }
    return blob;
}

void FcAfterConvTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    convParams convolutionParams;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(netPrecision, targetDevice, tempConfig, convolutionParams, inputChannels, outputChannels) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    std::vector<size_t> convInputShape = {1, inputChannels, 1, inputShape[0] * inputShape[1] / inputChannels};
    auto reshapePattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, convInputShape);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], reshapePattern1, false);

    auto filterWeights = ov::test::utils::generate_float_numbers(outputChannels * convInputShape[1] * kernelShape[0] * kernelShape[1],
                                                                 -0.1f, 0.1f);
    auto conv = ngraph::builder::makeConvolution(reshape1,
                                                 ngPrc,
                                                 {kernelShape[0], kernelShape[1]},
                                                 {kernelShape[0] > 1 ? stride : 1, stride},
                                                 {0, 0},
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, false, filterWeights);

    auto widthAfterConv = (convInputShape[3] - kernelShape[1]) / stride + 1;
    std::vector<size_t> outFormShapes = {1,  outputChannels * widthAfterConv };
    auto reshapePattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(conv, reshapePattern2, false);
    auto relu1 = std::make_shared<ngraph::opset3::Relu>(reshape2);

    std::vector<float> fc3_weights = ov::test::utils::generate_float_numbers(outFormShapes[1] * outFormShapes[1], -0.1f, 0.1f);
    auto fc3 = ngraph::builder::makeFullyConnected(relu1, ngPrc, outFormShapes[1], false, {}, fc3_weights);

    auto fc4_weights = ov::test::utils::generate_float_numbers(outFormShapes[1] * outFormShapes[1], -0.1f, 0.1f);
    auto fc4 = ngraph::builder::makeFullyConnected(fc3, ngPrc, outFormShapes[1], false, {}, fc4_weights);

    function = std::make_shared<ngraph::Function>(fc4, params, "FcAfterConvTest");
}

std::string FcBeforeConvTest::getTestCaseName(const testing::TestParamInfo<FcConvFcParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t inputChannels;
    size_t outputChannels;
    convParams convolutionParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(netPrecision, targetDevice, configuration, convolutionParams, inputChannels, outputChannels) = obj.param;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "KS=" << ov::test::utils::vec2str(kernelShape) << "_";
    result << "S=" << stride << "_";
    result << "IC=" << inputChannels << "_";
    result << "OC=" << outputChannels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr FcBeforeConvTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.1f, 0.1f);
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = values[i];
    }
    return blob;
}

void FcBeforeConvTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    convParams convolutionParams;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(netPrecision, targetDevice, tempConfig, convolutionParams, inputChannels, outputChannels) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto fc1_weights = ov::test::utils::generate_float_numbers(inputShape[1] * inputShape[1], -0.1f, 0.1f);
    auto fc1 = ngraph::builder::makeFullyConnected(params[0], ngPrc, inputShape[1], false, {}, fc1_weights);

    auto fc2_weights = ov::test::utils::generate_float_numbers(inputShape[1] * inputShape[1], -0.05f, 0.05f);
    auto fc2 = ngraph::builder::makeFullyConnected(fc1, ngPrc, inputShape[1], false, {}, fc2_weights);

    std::vector<size_t> convInputShape = {1, inputChannels, 1, inputShape[0] * inputShape[1] / inputChannels};
    auto reshapePattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, convInputShape);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(fc2, reshapePattern1, false);

    auto filterWeights = ov::test::utils::generate_float_numbers(outputChannels * convInputShape[1] * kernelShape[0] * kernelShape[1],
                                                                 -0.1f, 0.1f);
    auto conv = ngraph::builder::makeConvolution(reshape1,
                                                 ngPrc,
                                                 {kernelShape[0], kernelShape[1]},
                                                 {kernelShape[0] > 1 ? stride : 1, stride},
                                                 {0, 0},
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, false, filterWeights);

    auto widthAfterConv = (convInputShape[3] - kernelShape[1]) / stride + 1;
    std::vector<size_t> outFormShapes = {1,  outputChannels * widthAfterConv };
    auto reshapePattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(conv, reshapePattern2, false);

    function = std::make_shared<ngraph::Function>(reshape2, params, "FcBeforeConvTest");
}

std::string FcBetweenConvsTest::getTestCaseName(const testing::TestParamInfo<FcConvFcParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t inputChannels;
    size_t outputChannels;
    convParams convolutionParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(netPrecision, targetDevice, configuration, convolutionParams, inputChannels, outputChannels) = obj.param;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "KS=" << ov::test::utils::vec2str(kernelShape) << "_";
    result << "S=" << stride << "_";
    result << "IC=" << inputChannels << "_";
    result << "OC=" << outputChannels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr FcBetweenConvsTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = values[i];
    }
    return blob;
}

void FcBetweenConvsTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    convParams convolutionParams;
    size_t inputChannels;
    size_t outputChannels;
    std::tie(netPrecision, targetDevice, tempConfig, convolutionParams, inputChannels, outputChannels) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    std::vector<size_t> conv1InputShape = {1, inputChannels, 1, inputShape[0] * inputShape[1] / inputChannels};
    auto reshapePattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, conv1InputShape);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], reshapePattern1, false);

    auto filter1Weights = ov::test::utils::generate_float_numbers(outputChannels * conv1InputShape[1] * kernelShape[0] * kernelShape[1],
                                                                  -0.2f, 0.2f);
    auto conv1 = ngraph::builder::makeConvolution(reshape1,
                                                  ngPrc,
                                                  {kernelShape[0], kernelShape[1]},
                                                  {kernelShape[0] > 1 ? stride : 1, stride},
                                                  {0, 0},
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, false, filter1Weights);

    auto widthAfterConv1 = (conv1InputShape[3] - kernelShape[1]) / stride + 1;
    std::vector<size_t> outFormShapes1 = {1,  outputChannels * widthAfterConv1 };

    auto reshapePattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes1);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(conv1, reshapePattern2, false);
    auto relu = std::make_shared<ngraph::opset3::Relu>(reshape2);

    auto fc_weights = ov::test::utils::generate_float_numbers(outFormShapes1[1] * outFormShapes1[1], -0.2f, 0.2f);
    auto fc = ngraph::builder::makeFullyConnected(relu, ngPrc, outFormShapes1[1], false, {}, fc_weights);

    std::vector<size_t> conv2InputShape = {1, outputChannels, 1, widthAfterConv1};
    auto reshapePattern3 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, conv2InputShape);
    auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(fc, reshapePattern3, false);

    auto filter2Weights = ov::test::utils::generate_float_numbers(outputChannels * conv2InputShape[1],
                                                                  -0.2f, 0.2f);
    auto conv2 = ngraph::builder::makeConvolution(reshape3, ngPrc, { 1, 1 }, { 1, 1 }, { 0, 0 },
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, false, filter2Weights);
    std::vector<size_t> outFormShapes2 = {1,  outputChannels * conv2InputShape[3]};

    auto reshapePattern4 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
    auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(conv2, reshapePattern4, false);

    function = std::make_shared<ngraph::Function>(reshape4, params, "FcBetweenConvsTest");
}
}  // namespace SubgraphTestsDefinitions
