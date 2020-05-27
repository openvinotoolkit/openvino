// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/cum_sum.hpp"

namespace LayerTestsDefinitions {

std::string CumSumLayerTest::getTestCaseName(testing::TestParamInfo<cumSumParams> obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    int64_t axis;
    bool exclusive, reverse;
    std::string targetDevice;
    std::tie(inputShapes, inputPrecision, axis, exclusive, reverse, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Precision=" << inputPrecision.name() << "_";
    result << "Axis=" << axis << "_";
    result << "Exclusive=" << (exclusive ? "TRUE" : "FALSE") << "_";
    result << "Reverse=" << (reverse ? "TRUE" : "FALSE") << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void CumSumLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    bool exclusive, reverse;
    int64_t axis;
    std::tie(inputShapes, inputPrecision, axis, exclusive, reverse, targetDevice) = this->GetParam();
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    ngraph::ParameterVector paramVector;
    auto paramData = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape(inputShapes));
    paramVector.push_back(paramData);

    auto axisNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<int64_t>{axis})->output(0);

    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramVector));
    auto cumSum = std::dynamic_pointer_cast<ngraph::op::CumSum>(ngraph::builder::makeCumSum(paramOuts[0], axisNode, exclusive, reverse));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(cumSum)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "cumsum");
}

TEST_P(CumSumLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
