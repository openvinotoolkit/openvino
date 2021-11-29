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
    result << "axes=" << CommonTestUtils::vec2str(params.axes) << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void SliceLayerTest::SetUp() {
    SliceSpecificParams sliceParams;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additionalConfig;
    std::tie(sliceParams, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    element::Type_t et = element::i32;

    const auto data = std::make_shared<opset8::Parameter>(ngPrc, Shape(sliceParams.inputShape));
    const auto start = std::make_shared<opset8::Constant>(et, Shape{sliceParams.start.size()}, sliceParams.start);
    const auto stop = std::make_shared<opset8::Constant>(et, Shape{sliceParams.stop.size()}, sliceParams.stop);
    const auto step = std::make_shared<opset8::Constant>(et, Shape{sliceParams.step.size()}, sliceParams.step);

    Output<Node> slice;
    if (sliceParams.axes.empty()) {
        slice = std::make_shared<opset8::Slice>(data, start, stop, step);
    } else {
        const auto axes = std::make_shared<opset8::Constant>(et, Shape{sliceParams.axes.size()}, sliceParams.axes);
        slice = std::make_shared<opset8::Slice>(data, start, stop, step, axes);
    }

    ResultVector results{std::make_shared<opset8::Result>(slice)};
    function = std::make_shared<Function>(results, ov::ParameterVector{data}, "Slice");
}

}  // namespace LayerTestsDefinitions
