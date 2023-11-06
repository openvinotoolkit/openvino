// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/mvn_multiply_add.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace ov {
namespace test {

std::string MVNMultiplyAdd::getTestCaseName(const testing::TestParamInfo<mvnMultiplyAddParams>& obj) {
    std::pair<ov::Shape, ov::Shape> shapes;
    ov::Shape inputShapes, constantShapes;
    ov::element::Type dataPrecision, axesPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::string targetDevice;
    std::tie(shapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = obj.param;
    std::tie(inputShapes, constantShapes) = shapes;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "CS=" << ov::test::utils::vec2str(constantShapes) << "_";
    result << "DataET=" << dataPrecision << "_";
    result << "AxET=" << axesPrecision << "_";
    result << "Ax=" << ov::test::utils::vec2str(axes) << "_";
    result << "NormVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Eps=" << eps << "_";
    result << "EM=" << epsMode << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void MVNMultiplyAdd::SetUp() {
    std::pair<ov::Shape, ov::Shape> shapes;
    ov::Shape inputShapes, constantShapes;
    ov::element::Type dataType, axesType;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::tie(shapes, dataType, axesType, axes, normalizeVariance, eps, epsMode, targetDevice) = this->GetParam();
    std::tie(inputShapes, constantShapes) = shapes;

    ov::ParameterVector param{std::make_shared<ov::op::v0::Parameter>(dataType, ov::Shape(inputShapes))};
    auto axesNode = ngraph::builder::makeConstant(axesType, ov::Shape{axes.size()}, axes);
    auto mvn = ngraph::builder::makeMVN6(param[0], axesNode, normalizeVariance, eps, epsMode);
    auto gamma = ngraph::builder::makeConstant<float>(dataType, constantShapes, {}, true);
    auto mul = std::make_shared<ov::op::v1::Multiply>(mvn, gamma);
    auto beta = ngraph::builder::makeConstant<float>(dataType, constantShapes, {}, true);
    auto add = std::make_shared<ov::op::v1::Add>(mul, beta);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    function = std::make_shared<ov::Model>(results, param, "MVNMultiplyAdd");
}

}  // namespace test
}  // namespace ov
