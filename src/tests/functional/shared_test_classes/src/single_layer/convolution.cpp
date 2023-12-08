// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convolution.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj) {
    convSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) =
            obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ConvolutionLayerTest::SetUp() {
    convSpecificParams convParams;
    std::vector<size_t> inputShape;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
            this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    std::vector<float> filter_weights;
    if (targetDevice == ov::test::utils::DEVICE_GNA) {
        auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
        filter_weights = ov::test::utils::generate_float_numbers(convOutChannels * inputShape[1] * filter_size,
                                                                 -0.1f, 0.1f);
    }
    auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(
            ngraph::builder::makeConvolution(params[0], ngPrc, kernel, stride, padBegin,
                                             padEnd, dilation, padType, convOutChannels, false, filter_weights));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(conv)};
    function = std::make_shared<ngraph::Function>(results, params, "convolution");
}
}  // namespace LayerTestsDefinitions
