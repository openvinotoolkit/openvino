// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/parameter_result.hpp"

namespace SubgraphTestsDefinitions {

std::string ParameterResultSubgraphTestBase::getTestCaseName(const testing::TestParamInfo<parameterResultParams>& obj) {
    ov::test::InputShape inShape;
    std::string targetDevice;
    std::tie(inShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=";
    result << CommonTestUtils::partialShape2str({inShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : inShape.second) {
        result << CommonTestUtils::vec2str(shape) << "_";
    }
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

std::shared_ptr<ov::Model> ParameterResultSubgraphTestBase::createModel(const ov::PartialShape& shape) {
    auto parameter = std::make_shared<ngraph::opset1::Parameter>(ov::element::f32, shape);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(parameter)};
    ngraph::ParameterVector params = {parameter};
    auto model = std::make_shared<ov::Model>(results, params, "ParameterResult");
    return model;
}

void ParameterResultSubgraphTestLegacyApi::SetUp() {
    ov::test::InputShape inShape;
    std::tie(inShape, targetDevice) = this->GetParam();

    IE_ASSERT(inShape.first.is_static());

    function = createModel(inShape.first);
}

void ParameterResultSubgraphTest::SetUp() {
    ov::test::InputShape inShape;
    std::tie(inShape, targetDevice) = this->GetParam();

    init_input_shapes({inShape});

    function = createModel(inShape.first);
}

}  // namespace SubgraphTestsDefinitions
