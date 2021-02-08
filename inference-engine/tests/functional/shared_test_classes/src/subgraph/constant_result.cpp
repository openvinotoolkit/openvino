// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/constant_result.hpp"

namespace SubgraphTestsDefinitions {

std::string ConstantResultSubgraphTest::getTestCaseName(testing::TestParamInfo<constResultParams> obj) {
    std::string targetDevice;
    std::tie(targetDevice) = obj.param;
    std::ostringstream result;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ConstantResultSubgraphTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::tie(targetDevice) = this->GetParam();
    std::vector<float> data(300);
    for (size_t i = 0; i < 300; i++)
        data[i] = i;

    auto constant = std::make_shared<ngraph::opset5::Constant>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10}, data);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(constant)};
    ngraph::ParameterVector params;
    function = std::make_shared<ngraph::Function>(results, params, "ConstResult");
}

}  // namespace SubgraphTestsDefinitions

