// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/split.hpp"

namespace LayerTestsDefinitions {

std::string SplitLayerTest::getTestCaseName(const testing::TestParamInfo<splitParams>& obj) {
    size_t numSplits;
    int64_t axis;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, outIndices;
    std::string targetDevice;
    std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outIndices, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "numSplits=" << numSplits << "_";
    result << "axis=" << axis << "_";
    if (!outIndices.empty()) {
        result << "outIndices" << ov::test::utils::vec2str(outIndices) << "_";
    }
    result << "IS";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void SplitLayerTest::SetUp() {
    size_t axis, numSplits;
    std::vector<size_t> inputShape, outIndices;
    InferenceEngine::Precision netPrecision;
    std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outIndices, targetDevice) = this->GetParam();
    if (outIndices.empty()) {
        for (int i = 0; i < numSplits; ++i) {
            outIndices.push_back(i);
        }
    }
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto split = std::dynamic_pointer_cast<ngraph::opset5::Split>(ngraph::builder::makeSplit(params[0],
                                                                                             ngPrc, numSplits, axis));
    OPENVINO_SUPPRESS_DEPRECATED_END
    ngraph::ResultVector results;
    for (int i = 0; i < outIndices.size(); i++) {
        results.push_back(std::make_shared<ngraph::opset5::Result>(split->output(outIndices[i])));
    }
    function = std::make_shared<ngraph::Function>(results, params, "split");
}
}  // namespace LayerTestsDefinitions
