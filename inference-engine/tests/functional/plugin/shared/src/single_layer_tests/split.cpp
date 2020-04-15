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
    InferenceEngine::Precision inputPrecision, netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(numSplits, axis, inputPrecision, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "numSplits=" << numSplits << "_";
    result << "axis=" << axis << "_";
    result << "IS";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SplitLayerTest::SetUp() {
    size_t axis, numSplits;
    std::vector<size_t> inputShape;
    std::tie(numSplits, axis, inputPrecision, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto split = std::dynamic_pointer_cast<ngraph::opset1::Split>(ngraph::builder::makeSplit(paramOuts[0],
                                                                                             ngPrc, numSplits, axis));
    ngraph::ResultVector results;
    for (int i = 0; i < numSplits; i++) {
        results.push_back(std::make_shared<ngraph::opset1::Result>(split));
    }
    fnPtr = std::make_shared<ngraph::Function>(results, params, "split");
}

TEST_P(SplitLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions