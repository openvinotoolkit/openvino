// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionBackpropDataLayerTest::getTestCaseName(testing::TestParamInfo<convBackpropDataLayerTestParamsSet> obj) {
    convBackpropDataSpecificParams convBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convBackpropDataParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
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

void ConvolutionBackpropDataLayerTest::SetUp() {
    convBackpropDataSpecificParams convBackpropDataParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
            ngraph::builder::makeConvolutionBackpropData(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                         padEnd, dilation, padType, convOutChannels));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convBackpropData)};
    function = std::make_shared<ngraph::Function>(results, params, "convolutionBackpropData");
}

TEST_P(ConvolutionBackpropDataLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
