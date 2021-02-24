// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/convolution_relu_sequence.hpp"

namespace SubgraphTestsDefinitions {

std::string ConvolutionReluSequenceTest::getTestCaseName(testing::TestParamInfo<convReluSequenceTestParamsSet> obj) {
    convReluSpecificParamsAll convParamsAll;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    std::string targetDevice;
    std::tie(convParamsAll, netPrecision, inPrc, outPrc, targetDevice) =
        obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(convParamsAll.inputShape) << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice << "_";

    for (auto&& single : convParamsAll.sequenceDesc) {
        result << "K" << CommonTestUtils::vec2str(single.kernelSize) << "_";
        result << "S" << CommonTestUtils::vec2str(single.strides) << "_";
        result << "PB" << CommonTestUtils::vec2str(single.padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(single.padEnd) << "_";
        result << "O=" << single.numOutChannels << "_";
    }

    return result.str();
}

void ConvolutionReluSequenceTest::SetUp() {
    threshold = 0.0031;
    const InferenceEngine::SizeVector dilation = { 1, 1 };
    convReluSpecificParamsAll convParamsAll;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convParamsAll, netPrecision, inPrc, outPrc, targetDevice) =
        this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, { convParamsAll.inputShape});
    auto lastOutputs = ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params).front();
    auto inputChannels = convParamsAll.inputShape[1];

    for (auto&& single : convParamsAll.sequenceDesc) {
        const auto addBiases = true;
        const auto filtersRange = 0.1f;
        const auto biasesRange = 0.05f;
        std::vector<float> filter_weights;
        std::vector<float> biases;
        if (targetDevice == CommonTestUtils::DEVICE_GNA) {
            auto filter_size = std::accumulate(std::begin(single.kernelSize), std::end(single.kernelSize), 1, std::multiplies<size_t>());
            filter_weights = CommonTestUtils::generate_float_numbers(single.numOutChannels * inputChannels * filter_size,
                -filtersRange, filtersRange);
            if (addBiases) {
                biases = CommonTestUtils::generate_float_numbers(single.numOutChannels,
                    -biasesRange, biasesRange);
            }
        }

        std::shared_ptr<ngraph::Node> conv =
            std::dynamic_pointer_cast<ngraph::Node>(
                ngraph::builder::makeConvolution(
                    lastOutputs,
                    ngPrc, single.kernelSize, single.strides, single.padBegin, single.padEnd,
                    dilation, ngraph::op::PadType::EXPLICIT, single.numOutChannels, addBiases, filter_weights, biases));
        lastOutputs = std::make_shared<ngraph::opset1::Relu>(conv);
        inputChannels = single.numOutChannels;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lastOutputs)};
    function = std::make_shared<ngraph::Function>(results, params, "convolution_relu_sequence");
}
}  // namespace SubgraphTestsDefinitions
