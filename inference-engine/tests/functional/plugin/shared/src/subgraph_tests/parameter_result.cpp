// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/parameter_result.hpp"

namespace LayerTestsDefinitions {

std::string ParameterResultSubgraphTest::getTestCaseName(testing::TestParamInfo<paramResultParams> obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::string targetDevice;
    std::tie(inputPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Precision=" << inputPrecision.name() << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ParameterResultSubgraphTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::tie(inputPrecision, inputShapes, targetDevice) = this->GetParam();
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto param = ngraph::builder::makeParams(inType, {inputShapes});
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(param[0])};
    function = std::make_shared<ngraph::Function>(results, param, "ReluShapeOf");
}

TEST_P(ParameterResultSubgraphTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions

