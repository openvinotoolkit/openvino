// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/parameter_result.hpp"

namespace ov {
namespace test {

std::string ParameterResultSubgraphTest::getTestCaseName(const testing::TestParamInfo<parameterResultParams>& obj) {
    ov::test::InputShape inShape;
    std::string targetDevice;
    std::tie(inShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=";
    result << ov::test::utils::partialShape2str({inShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : inShape.second) {
        result << ov::test::utils::vec2str(shape) << "_";
    }
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

std::shared_ptr<ov::Model> ParameterResultSubgraphTest::createModel(const ov::PartialShape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(parameter)};
    ov::ParameterVector params = {parameter};
    auto model = std::make_shared<ov::Model>(results, params, "ParameterResult");
    return model;
}

void ParameterResultSubgraphTest::SetUp() {
    ov::test::InputShape inShape;
    std::tie(inShape, targetDevice) = this->GetParam();

    init_input_shapes({inShape});

    function = createModel(inShape.first);
}

}  // namespace test
}  // namespace ov
