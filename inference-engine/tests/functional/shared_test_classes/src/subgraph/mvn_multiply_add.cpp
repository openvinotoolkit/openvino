// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/mvn_multiply_add.hpp"

namespace SubgraphTestsDefinitions {

std::string MVNMultiplyAdd::getTestCaseName(const testing::TestParamInfo<mvnMultiplyAddParams> &obj) {
    std::pair<InferenceEngine::SizeVector, InferenceEngine::SizeVector> shapes;
    InferenceEngine::SizeVector inputShapes, constantShapes;
    InferenceEngine::Precision dataPrecision, axesPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::string targetDevice;
    std::tie(shapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = obj.param;
    std::tie(inputShapes, constantShapes) = shapes;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "CS=" << CommonTestUtils::vec2str(constantShapes) << "_";
    result << "DataPrc=" << dataPrecision.name() << "_";
    result << "AxPrc=" << axesPrecision.name() << "_";
    result << "Ax=" << CommonTestUtils::vec2str(axes) << "_";
    result << "NormVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Eps=" << eps << "_";
    result << "EM=" << epsMode << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void MVNMultiplyAdd::SetUp() {
    std::pair<InferenceEngine::SizeVector, InferenceEngine::SizeVector> shapes;
    InferenceEngine::SizeVector inputShapes, constantShapes;
    InferenceEngine::Precision dataPrecision, axesPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::tie(shapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = this->GetParam();
    std::tie(inputShapes, constantShapes) = shapes;

    auto dataType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrecision);
    auto axesType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(axesPrecision);

    auto param = ngraph::builder::makeParams(dataType, {inputShapes});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
    auto axesNode = ngraph::builder::makeConstant(axesType, ngraph::Shape{axes.size()}, axes);
    auto mvn = ngraph::builder::makeMVN6(paramOuts[0], axesNode, normalizeVariance, eps, epsMode);
    auto gamma = ngraph::builder::makeConstant<float>(dataType, constantShapes, {}, true);
    auto mul = std::make_shared<ngraph::opset7::Multiply>(mvn, gamma);
    auto beta = ngraph::builder::makeConstant<float>(dataType, constantShapes, {}, true);
    auto add = std::make_shared<ngraph::opset7::Add>(mul, beta);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
    function = std::make_shared<ngraph::Function>(results, param, "MVNMultiplyAdd");
}
} // namespace SubgraphTestsDefinitions
