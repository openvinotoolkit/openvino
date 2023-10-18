// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/input_conv.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string InputConvTest::getTestCaseName(const testing::TestParamInfo<inputConvParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t outputChannels;
    convParams convolutionParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    bool addReshape;
    std::tie(netPrecision, targetDevice, configuration, convolutionParams, outputChannels, addReshape) = obj.param;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "KS=" << ov::test::utils::vec2str(kernelShape) << "_";
    result << "S=" << stride << "_";
    result << "OC=" << outputChannels << "_";
    result << "addReshape=" << addReshape << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr InputConvTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();
    auto precision = info.getPrecision();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    for (size_t i = 0; i < blob->size(); i++) {
        float value = i % 16;
        if (typeid(precision) == typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = ngraph::float16(value).to_bits();
        } else {
            rawBlobDataPtr[i] = value;
        }
    }
    return blob;
}

void InputConvTest::SetUp() {
    auto generateWeights = [](std::size_t out_channels, std::size_t kernel_size) {
        std::vector<float> res;
        for (std::size_t i = 0; i < out_channels; ++i) {
            for (std::size_t j = 0; j < kernel_size; ++j) {
                j == 0 ? res.emplace_back(0.2f) : res.emplace_back(0.0f);
            }
        }

        return res;
    };

    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    convParams convolutionParams;
    size_t outputChannels;
    bool addReshape;
    std::tie(netPrecision, targetDevice, tempConfig, convolutionParams, outputChannels, addReshape) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto conv0 = ngraph::builder::makeConvolution(params[0],
                                                  ngPrc,
                                                  {kernelShape[0], kernelShape[1]},
                                                  {kernelShape[0] > 1 ? stride : 1, stride},
                                                  {0, 0},
                                                  {0, 0},
                                                  {1, 1},
                                                  ngraph::op::PadType::VALID,
                                                  outputChannels,
                                                  true,
                                                  generateWeights(outputChannels, kernelShape[1]));

    if (addReshape) {
        size_t numOutputWidth = (((inputShape[1] * inputShape[2] * inputShape[3] - kernelShape[1] * kernelShape[0]) / (inputShape[1] * stride)) + 1);
        std::vector<size_t> outFormShapes0 = { 1, outputChannels * numOutputWidth };
        auto pattern0 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes0);
        auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(conv0, pattern0, false);

        ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(reshape0) };
        function = std::make_shared<ngraph::Function>(results, params, "InputConvTest");
    } else {
        ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(conv0) };
        function = std::make_shared<ngraph::Function>(results, params, "InputConvTest");
    }
}
}  // namespace SubgraphTestsDefinitions
