// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/binary_convolution.hpp"

namespace LayerTestsDefinitions {

std::string BinaryConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<binaryConvolutionTestParamsSet>& obj) {
    binConvSpecificParams binConvParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;

    std::tie(binConvParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = obj.param;

    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    float padValue;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, padValue) = binConvParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "KS=" << ov::test::utils::vec2str(kernel) << "_";
    result << "S=" << ov::test::utils::vec2str(stride) << "_";
    result << "PB=" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE=" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "PV=" << padValue << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr BinaryConvolutionLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::Blob::Ptr blobPtr;
    const std::string name = info.name();
    // there is no input generation for filters since CPU implementation uses Constant
    // TODO: enable filters input generation as Parameter when supported (Issue 50148)
    if (name == "a_data_batch") {
        blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 1, 0, 1, 7235346);
    }
    return blobPtr;
}

void BinaryConvolutionLayerTest::SetUp() {
    binConvSpecificParams binConvParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;

    std::tie(binConvParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
        this->GetParam();

    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernelSize, strides, dilations;
    std::vector<ptrdiff_t> padsBegin, padsEnd;
    size_t numOutChannels;
    float padValue;
    std::tie(kernelSize, strides, padsBegin, padsEnd, dilations, numOutChannels, padType, padValue) = binConvParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params[0]->set_friendly_name("a_data_batch");

    // TODO: refactor build BinaryConvolution op to accept filters input as Parameter
    auto binConv = ngraph::builder::makeBinaryConvolution(params[0], kernelSize, strides, padsBegin, padsEnd, dilations, padType, numOutChannels,
                                                          padValue);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(binConv)};
    function = std::make_shared<ngraph::Function>(results, params, "BinaryConvolution");
}

}   // namespace LayerTestsDefinitions
