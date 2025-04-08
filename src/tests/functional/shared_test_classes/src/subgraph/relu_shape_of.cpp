// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/relu_shape_of.hpp"

namespace ov {
namespace test {

std::string ReluShapeOfSubgraphTest::getTestCaseName(const testing::TestParamInfo<ov::test::ShapeOfParams>& obj) {
    ov::Shape inputShapes;
    ov::element::Type element_type, output_type;
    std::string targetDevice;
    std::tie(element_type, output_type, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "IET=" << element_type << "_";
    result << "OET=" << output_type << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void ReluShapeOfSubgraphTest::SetUp() {
    ov::Shape inputShapes;
    ov::element::Type element_type, output_type;
    std::tie(element_type, output_type, inputShapes, targetDevice) = this->GetParam();
    ov::ParameterVector param{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(inputShapes))};
    auto relu = std::make_shared<ov::op::v0::Relu>(param[0]);
    auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(relu, output_type);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(shapeOf)};
    function = std::make_shared<ov::Model>(results, param, "ReluShapeOf");
}

}  // namespace test
}  // namespace ov
