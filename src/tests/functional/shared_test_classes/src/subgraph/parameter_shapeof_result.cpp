// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_ngraph_utils.hpp>

#include "shared_test_classes/subgraph/parameter_shapeof_result.hpp"

namespace SubgraphTestsDefinitions {

std::string ParameterShapeOfResultSubgraphTest::getTestCaseName(const testing::TestParamInfo<parameterShapeOfResultParams>& obj) {
    ov::element::Type inType;
    std::string targetDevice;
    std::tie(inType, targetDevice) = obj.param;
    std::ostringstream result;
    result << "InType=" << inType;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ParameterShapeOfResultSubgraphTest::SetUp() {
    ov::element::Type inType;
    std::tie(inType, targetDevice) = this->GetParam();
    inPrc = InferenceEngine::details::convertPrecision(inType);

    const auto parameter = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape{1, 3, 10, 10});
    const auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(parameter);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(shapeOf)};
    ov::ParameterVector params = {parameter};
    function = std::make_shared<ov::Model>(results, params, "ParameterShapeOfResult");
}

}  // namespace SubgraphTestsDefinitions
