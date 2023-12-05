// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mvn.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

std::string Mvn1LayerTest::getTestCaseName(const testing::TestParamInfo<mvn1Params>& obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    ngraph::AxisSet axes;
    bool acrossChannels, normalizeVariance;
    double eps;
    std::string targetDevice;
    std::tie(inputShapes, inputPrecision, axes, acrossChannels, normalizeVariance, eps, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "Precision=" << inputPrecision.name() << "_";
    if (!axes.empty()) {
        result << "ReductionAxes=" << ov::test::utils::vec2str(axes.to_vector()) << "_";
    } else {
        result << "AcrossChannels=" << (acrossChannels ? "TRUE" : "FALSE") << "_";
    }
    result << "NormalizeVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Epsilon=" << eps << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void Mvn1LayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    ngraph::AxisSet axes;
    bool acrossChanels, normalizeVariance;
    double eps;
    std::tie(inputShapes, inputPrecision, axes, acrossChanels, normalizeVariance, eps, targetDevice) = this->GetParam();
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    ov::ParameterVector param {std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(inputShapes))};
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto mvn = std::dynamic_pointer_cast<ngraph::op::MVN>(ngraph::builder::makeMVN(param[0], acrossChanels, normalizeVariance, eps));
    if (!axes.empty()) {
        mvn = std::dynamic_pointer_cast<ngraph::op::MVN>(ngraph::builder::makeMVN(param[0], axes, normalizeVariance, eps));
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn)};
    function = std::make_shared<ngraph::Function>(results, param, "MVN1");
}


std::string Mvn6LayerTest::getTestCaseName(const testing::TestParamInfo<mvn6Params>& obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision, axesPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::string targetDevice;
    std::tie(inputShapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "DataPrc=" << dataPrecision.name() << "_";
    result << "AxPrc=" << axesPrecision.name() << "_";
    result << "Ax=" << ov::test::utils::vec2str(axes) << "_";
    result << "NormVariance=" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "Eps=" << eps << "_";
    result << "EM=" << epsMode << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void Mvn6LayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision dataPrecision, axesPrecision;
    std::vector<int> axes;
    bool normalizeVariance;
    float eps;
    std::string epsMode;
    std::tie(inputShapes, dataPrecision, axesPrecision, axes, normalizeVariance, eps, epsMode, targetDevice) = this->GetParam();

    auto dataType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dataPrecision);
    auto axesType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(axesPrecision);

    ov::ParameterVector param {std::make_shared<ov::op::v0::Parameter>(dataType, ov::Shape(inputShapes))};
    auto axesNode = ngraph::builder::makeConstant(axesType, ngraph::Shape{axes.size()}, axes);
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto mvn = ngraph::builder::makeMVN6(param[0], axesNode, normalizeVariance, eps, epsMode);
    OPENVINO_SUPPRESS_DEPRECATED_END
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn)};
    function = std::make_shared<ngraph::Function>(results, param, "MVN6");
}

}  // namespace LayerTestsDefinitions
