// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "ngraph/ngraph.hpp"

#include "shared_test_classes/single_layer/slice.hpp"

using namespace ngraph;

namespace LayerTestsDefinitions {

std::string SliceLayerTest::getTestCaseName(const testing::TestParamInfo<SliceParams> &obj) {
    SliceSpecificParams params;
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(params, netPrc, inPrc, outPrc, inLayout, outLayout, targetName, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "inShape=" << CommonTestUtils::vec2str(params.inputShape) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "start=" << CommonTestUtils::vec2str(params.start) << "_";
    result << "stop=" << CommonTestUtils::vec2str(params.stop) << "_";
    result << "step=" << CommonTestUtils::vec2str(params.step) << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void SliceLayerTest::SetUp() {
    SliceSpecificParams ssParams;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additionalConfig;
    std::tie(ssParams, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    element::Type_t et = element::i32;

    const auto data = std::make_shared<opset8::Parameter>(ngPrc, Shape(ssParams.inputShape));
    const auto start = std::make_shared<opset8::Constant>(et, Shape{ssParams.start.size()}, ssParams.start);
    const auto stop = std::make_shared<opset8::Constant>(et, Shape{ssParams.stop.size()}, ssParams.stop);
    const auto step = std::make_shared<opset8::Constant>(et, Shape{ssParams.step.size()}, ssParams.step);

    const auto axes = std::make_shared<opset8::Constant>(et, Shape{ssParams.axes.size()}, ssParams.axes);

    auto ss = std::make_shared<opset8::Slice>(data, start, stop, step, axes);
    ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(ss)};
    function = std::make_shared<ngraph::Function>(results, ov::ParameterVector{data}, "Slice");
}

}  // namespace LayerTestsDefinitions
