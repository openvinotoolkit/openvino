// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/concat_conv.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string ConcatConvTest::getTestCaseName(const testing::TestParamInfo<ConcatConvParams>& obj) {
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

InferenceEngine::Blob::Ptr ConcatConvTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -2.0f, 2.0f);
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = values[i];
    }
    return blob;
}

void ConcatConvTest::SetUp() {
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
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(params[0]);

    auto const_values = ov::test::utils::generate_float_numbers(inputShape[1], -2.0f, 2.0f);
    auto constant = ngraph::builder::makeConstant(ngPrc, inputShape, const_values);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{constant, relu1}, 1);

    std::vector<size_t> convInputShape = {1, inputChannels, 1, 2 * inputShape[0] * inputShape[1] / inputChannels};
    auto reshapePattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, convInputShape);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(concat, reshapePattern1, false);

    auto filterWeights = ov::test::utils::generate_float_numbers(outputChannels * convInputShape[1] * kernelShape[0] * kernelShape[1],
                                                                 -0.2f, 0.2f);
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

    function = std::make_shared<ngraph::Function>(reshape2, params, "ConcatConvTest");
}

}  // namespace SubgraphTestsDefinitions
