// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiply_add.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace ov {
namespace test {

std::string MultiplyAddLayerTest::getTestCaseName(const testing::TestParamInfo<MultiplyAddParamsTuple>& obj) {
    ov::Shape inputShapes;
    ov::element::Type element_type;
    std::string targetName;
    std::tie(inputShapes, element_type, targetName) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    results << "ET=" << element_type << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void MultiplyAddLayerTest::SetUp() {
    ov::Shape inputShape;
    ov::element::Type element_type;
    std::tie(inputShape, element_type, targetDevice) = this->GetParam();
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape(inputShape))};

    std::vector<size_t> constShape(inputShape.size(), 1);
    constShape[1] = inputShape[1];

    auto const_mul = ngraph::builder::makeConstant<float>(element_type, constShape, {}, true);
    auto mul = std::make_shared<ov::op::v1::Multiply>(params[0], const_mul);
    auto const_add = ngraph::builder::makeConstant<float>(element_type, constShape, {}, true);
    auto add = std::make_shared<ov::op::v1::Add>(mul, const_add);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    function = std::make_shared<ov::Model>(results, params, "multiplyAdd");
}

}  // namespace test
}  // namespace ov
