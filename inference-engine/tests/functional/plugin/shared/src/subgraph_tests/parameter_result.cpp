// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/parameter_result.hpp"

namespace LayerTestsDefinitions {

std::string ParameterResultSubgraphTest::getTestCaseName(testing::TestParamInfo<parameterResultParams> obj) {
    std::string targetDevice;
    std::tie(targetDevice) = obj.param;
    std::ostringstream result;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ParameterResultSubgraphTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::tie(targetDevice) = this->GetParam();

    auto parameter = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(parameter)};
    ngraph::ParameterVector params = {parameter};
    function = std::make_shared<ngraph::Function>(results, params, "ParameterResult");
}

TEST_P(ParameterResultSubgraphTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
