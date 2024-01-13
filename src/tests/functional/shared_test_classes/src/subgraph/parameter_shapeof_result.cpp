// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/parameter_shapeof_result.hpp"

#include <ie_ngraph_utils.hpp>

namespace SubgraphTestsDefinitions {

std::string ParameterShapeOfResultSubgraphTest::getTestCaseName(const testing::TestParamInfo<parameterShapeOfResultParams>& obj) {
    ngraph::element::Type inType;
    std::string targetDevice;
    std::tie(inType, targetDevice) = obj.param;
    std::ostringstream result;
    result << "InType=" << inType;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ParameterShapeOfResultSubgraphTest::SetUp() {
    ngraph::element::Type inType;
    std::tie(inType, targetDevice) = this->GetParam();
    inPrc = InferenceEngine::details::convertPrecision(inType);

    const auto parameter = std::make_shared<ov::op::v0::Parameter>(inType, ngraph::Shape{1, 3, 10, 10});
    const auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(parameter);
    const ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(shapeOf)};
    ngraph::ParameterVector params = {parameter};
    function = std::make_shared<ngraph::Function>(results, params, "ParameterShapeOfResult");
}

}  // namespace SubgraphTestsDefinitions
