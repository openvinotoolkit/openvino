// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/split.hpp"

namespace LayerTestsDefinitions {

std::string SplitLayerTest::getTestCaseName(testing::TestParamInfo<splitParams> obj) {
    size_t numSplits, axis;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "numSplits=" << numSplits << "_";
    result << "axis=" << axis << "_";
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
    SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
    size_t axis, numSplits;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto split = std::dynamic_pointer_cast<ngraph::opset1::Split>(ngraph::builder::makeSplit(paramOuts[0],
                                                                                             ngPrc, numSplits, axis));
    ngraph::ResultVector results;
    for (int i = 0; i < numSplits; i++) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(i)));
    }
    function = std::make_shared<ngraph::Function>(results, params, "split");
}

TEST_P(SplitLayerTest, CompareWithRefs) {
    Run();
};

std::string splitWithDiffOutsTest::getTestCaseName(testing::TestParamInfo<splitWithDiffOutsParams> obj) {
    size_t numSplits, axis;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::vector<size_t> outIndices;
    std::string targetDevice;
    std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outIndices, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "numSplits=" << numSplits << "_";
    result << "axis=" << axis << "_";
    result << "outIndices" << CommonTestUtils::vec2str(outIndices) << "_";
    result << "IS";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void splitWithDiffOutsTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
    size_t axis, numSplits;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> outIndeces;
    std::tie(numSplits, axis, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outIndeces, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto split = std::dynamic_pointer_cast<ngraph::opset1::Split>(ngraph::builder::makeSplit(paramOuts[0],
                                                                                             ngPrc, numSplits, axis));
    ngraph::ResultVector results;
    for (int i = 0; i < outIndeces.size(); i++) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split->output(outIndeces[i])));
    }
    function = std::make_shared<ngraph::Function>(results, params, "split");
}

TEST_P(splitWithDiffOutsTest, CompareWithRefs) {
    Run();
};


}  // namespace LayerTestsDefinitions