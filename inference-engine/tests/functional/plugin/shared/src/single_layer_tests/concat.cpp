// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatLayerTest::getTestCaseName(const testing::TestParamInfo<concatParamsTuple> &obj) {
    size_t axis;
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision, inputPrecision;
    std::string targetName;
    std::tie(axis, inputShapes, inputPrecision, netPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "axis=" << axis << "_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void ConcatLayerTest::SetUp() {
    size_t axis;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(axis, inputShape, inputPrecision, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, inputShape);
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "concat");
}


TEST_P(ConcatLayerTest, CompareWithRefs) {
    inferAndValidate();
};
}  // namespace LayerTestsDefinitions