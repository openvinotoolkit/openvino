// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/convolution_relu_sequence.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

namespace SubgraphTestsDefinitions {

std::string ConvolutionReluSequenceTest::getTestCaseName(const testing::TestParamInfo<convReluSequenceTestParamsSet>& obj) {
    convReluSpecificParamsAll convParamsAll;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(convParamsAll, netPrecision, inPrc, outPrc, targetDevice, config) =
        obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(convParamsAll.inputShape) << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice << "_";

    for (auto&& single : convParamsAll.sequenceDesc) {
        result << "K" << ov::test::utils::vec2str(single.kernelSize) << "_";
        result << "S" << ov::test::utils::vec2str(single.strides) << "_";
        result << "PB" << ov::test::utils::vec2str(single.padBegin) << "_";
        result << "PE" << ov::test::utils::vec2str(single.padEnd) << "_";
        result << "O=" << single.numOutChannels << "_";
        result << "PW" << ov::test::utils::vec2str(single.poolingWindow) << "_";
        result << "PS" << ov::test::utils::vec2str(single.poolingStride) << "_";
    }

    for (auto&& single : config) {
        result << single.first << "=" << single.second;
    }
    return result.str();
}

void ConvolutionReluSequenceTest::SetUp() {
    threshold = 0.0031;
    const InferenceEngine::SizeVector dilation = { 1, 1 };
    convReluSpecificParamsAll convParamsAll;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::map<std::string, std::string> config;
    std::tie(convParamsAll, netPrecision, inPrc, outPrc, targetDevice, config) =
        this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(convParamsAll.inputShape))};
    std::shared_ptr<ov::Node> lastOutputs = params.front();
    auto inputChannels = convParamsAll.inputShape[1];

    for (auto&& single : convParamsAll.sequenceDesc) {
        const auto addBiases = true;
        const auto filtersRange = 0.1f;
        const auto biasesRange = 0.05f;
        std::vector<float> filter_weights;
        std::vector<float> biases;

        std::shared_ptr<ngraph::Node> conv =
            std::dynamic_pointer_cast<ngraph::Node>(
                ov::test::utils::make_convolution(
                    lastOutputs,
                    ngPrc, single.kernelSize, single.strides, single.padBegin, single.padEnd,
                    dilation, ov::op::PadType::EXPLICIT, single.numOutChannels, addBiases, filter_weights, biases));
        lastOutputs = std::make_shared<ov::op::v0::Relu>(conv);
        if (single.poolingWindow.size() == 2 &&
                (single.poolingWindow[0] != 1 ||
                 single.poolingWindow[1] != 1)) {
            lastOutputs = std::make_shared<ov::op::v1::MaxPool>(lastOutputs, single.poolingStride,
                ngraph::Shape{ 0, 0 },
                ngraph::Shape{ 0, 0 },
                single.poolingWindow);
        }
        inputChannels = single.numOutChannels;
    }

    ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(lastOutputs)};
    function = std::make_shared<ngraph::Function>(results, params, "convolution_relu_sequence");
}
}  // namespace SubgraphTestsDefinitions
