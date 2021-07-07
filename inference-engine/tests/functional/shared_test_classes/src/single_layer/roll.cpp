// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/roll.hpp"

namespace LayerTestsDefinitions {

std::string RollLayerTest::getTestCaseName(testing::TestParamInfo<rollParams> obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::vector<int64_t> shift;
    std::vector<int64_t> axes;
    std::string targetDevice;
    std::tie(inputShapes, inputPrecision, shift, axes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Precision=" << inputPrecision.name() << "_";
    result << "Shift=" << CommonTestUtils::vec2str(shift) << "_";
    result << "Axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void RollLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision inputPrecision;
    std::vector<int64_t> shift;
    std::vector<int64_t> axes;
    std::tie(inputShapes, inputPrecision, shift, axes, targetDevice) = this->GetParam();
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    ngraph::ParameterVector paramVector;
    auto paramData = std::make_shared<ngraph::opset1::Parameter>(inType, ngraph::Shape(inputShapes));
    paramVector.push_back(paramData);

    auto shiftNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{shift.size()}, shift)->output(0);
    auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{axes.size()}, axes)->output(0);

    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramVector));
    auto roll = std::dynamic_pointer_cast<ngraph::op::v7::Roll>(ngraph::builder::makeRoll(paramOuts[0], shiftNode, axesNode));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(roll)};
    function = std::make_shared<ngraph::Function>(results, paramVector, "roll");
}
}  // namespace LayerTestsDefinitions
