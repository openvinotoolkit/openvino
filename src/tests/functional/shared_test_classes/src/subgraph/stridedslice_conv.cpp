// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/stridedslice_conv.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string SliceConvTest::getTestCaseName(const testing::TestParamInfo<SliceConvParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t outputChannels;
    convParams convolutionParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(netPrecision, targetDevice, configuration, convolutionParams, outputChannels) = obj.param;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "KS=" << ov::test::utils::vec2str(kernelShape) << "_";
    result << "S=" << stride << "_";
    result << "OC=" << outputChannels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr SliceConvTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -2.0f, 2.0f);
    for (size_t i = 0; i < blob->size(); i++) {
        rawBlobDataPtr[i] = values[i];
    }
    return blob;
}

void SliceConvTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    convParams convolutionParams;
    size_t outputChannels;
    std::tie(netPrecision, targetDevice, tempConfig, convolutionParams, outputChannels) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto ss = ngraph::builder::makeStridedSlice(params[0], std::vector<int64_t>{0, 0, 0, 64}, std::vector<int64_t>{1, 1, 1, 128},
                                                std::vector<int64_t>{1, 1, 1, 1}, ngPrc, std::vector<int64_t>{1, 1, 1, 0},
                                                std::vector<int64_t>{1, 1, 1, 0}, std::vector<int64_t>{0, 0, 0, 0},
                                                std::vector<int64_t>{0, 0, 0, 0}, std::vector<int64_t>{0, 0, 0, 0});

    auto filterWeights = ov::test::utils::generate_float_numbers(outputChannels * inputShape[1] * kernelShape[0] * kernelShape[1],
                                                                 -0.2f, 0.2f);
    auto conv = ngraph::builder::makeConvolution(ss,
                                                 ngPrc,
                                                 {kernelShape[0], kernelShape[1]},
                                                 {kernelShape[0] > 1 ? stride : 1, stride},
                                                 {0, 0},
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, false, filterWeights);

    function = std::make_shared<ngraph::Function>(conv, params, "StridedSliceConvTest");
}

}  // namespace SubgraphTestsDefinitions
